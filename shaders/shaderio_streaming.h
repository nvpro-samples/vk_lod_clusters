/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio_scene.h"

#ifndef _SHADERIO_STREAMING_H_
#define _SHADERIO_STREAMING_H_

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else
// only affect shaders

// light weight check for clas allocator, counts number of used bits
// during build_freegaps and compares it against persistently tracked
// allocatedSize
#define STREAMING_DEBUG_USEDBITS_COUNT 1

// heavy weight, tests all bits being 0 or 1 respectively for the space
// a group takes on load, or unload.
#define STREAMING_DEBUG_FREEGAPS_OVERLAP 0

// hit streaming statistics binned by lod-level (written by
// stream_hit_propagation.comp.glsl into the readback debug arrays). bins are
// relative to each geometry's coarsest lod level (bin 0 = coarsest):
//   debugA = resident groups hit within 32 frames, debugB = resident groups,
//   debugC = total groups (independent of residency)
#define STREAM_DEBUG_HIT_STATS 0

// resident clusters binned by why they are kept (written by stream_agefilter_groups.comp.glsl
// into debugA, debugB counts groups): 0 used by traversal and hit within 2 frames,
// 1 used and hit within 3..32, 2 used and hit older, 3 unused within the unload delay,
// 4 kept by blas caching
#define STREAM_DEBUG_RESIDENCY 0

#endif

/////////////////////////////////////////

// we tag the 64 bit VA of a cluster group with a special address range
// when it hasn't been loaded yet, values greater than this encode the
// frame index that this group has been requested last.
#define STREAMING_INVALID_ADDRESS_START (uint64_t(1) << 63)

// checks previous address when loading/unloading a group
#define STREAMING_DEBUG_ADDRESSES 0

// simplifies debugging clas allocator by doing full build of free lists
// every frame
#define STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS 0

// disables cluster move operations and blas/tlas builds as as well as actual ray tracing
#define STREAMING_DEBUG_WITHOUT_RT 0

// Must be 32 to se we have easier processing of the bit arrays.
// the minimum allocation will use 32 bits, which can only straddle across two u32 values
#define STREAMING_ALLOCATOR_MIN_SIZE 32

// must fit in uint16_t
#define STREAMING_CACHED_BLAS_MAX_CLUSTERS 0xFFFF

// hit tracking packs the per-group hit state into a single uint32:
//
//   bit  31     STREAMING_HITFRAME_NO_FINER_BIT, set when no finer (generating)
//               group is resident. Inverted polarity: hit registration is a
//               single atomicMax that always sets it, so a fresh hit wins; the
//               flag only matters for stale frames and stream_hit_propagation
//               rewrites frame and flag together every frame.
//   bits 30..16 cyclic frame index of the last hit (15 bit). Consumers compare
//               via cyclic deltas; the propagation pass re-normalizes stamps past
//               the current frame so the atomicMax ordering survives the wrap.
//   bits 15..0  ray-cone width in object space, inverted fixed-point log2 (larger
//               = smaller width), so atomicMax yields "newest frame, then smallest
//               width". Only with USE_HIT_STREAMING_LOD; plain writes zero.
#define STREAMING_HITFRAME_NO_FINER_BIT (1u << 31)
#define STREAMING_HITFRAME_FRAME_SHIFT 16
#define STREAMING_HITFRAME_FRAME_MASK 0x7FFFu
#define STREAMING_HITFRAME_WIDTH_MASK 0xFFFFu
// The frame stamp cycles through BASE..MASK. 0 is "never": zero-filled buffers and entries
// older than BASE frames. The propagation re-stamps resident entries carried over the wrap
// below BASE.
#define STREAMING_HITFRAME_FRAME_BASE 0x1000u
#define STREAMING_HITFRAME_PERIOD (STREAMING_HITFRAME_FRAME_MASK + 1 - STREAMING_HITFRAME_FRAME_BASE)
#define STREAMING_HITFRAME_CYCLIC(frameIndex) (STREAMING_HITFRAME_FRAME_BASE + (frameIndex) % STREAMING_HITFRAME_PERIOD)
// the demand descent only re-stamps the groups it visits, so every geometry's `groupDemand` is
// cleared once within this many frames, before an old stamp could alias as fresh
#define STREAMING_HITFRAME_DEMAND_CLEAR_FRAMES 8192

// fixed-point log2 encoding range of the object-space hit cone width,
// covers [2^-32, 2^32) with ~0.1% steps
#define STREAMING_HITLOD_WIDTH_LOG2_MIN (-32.0f)
#define STREAMING_HITLOD_WIDTH_LOG2_SCALE (65535.0f / 64.0f)

// recency window (in frames) within which a hit entry drives the lod demand
// tests. Must be larger than the per-lod-level lag of the hit propagation pass
// (one level per frame) and smaller than the hit-unload threshold.
#define STREAMING_HITLOD_DEMAND_WINDOW 32

// The demand BFS (`stream_hit_demand_tree.comp.glsl`) does not re-stamp every demanded
// group every frame: during descent it stops at any finer group still fresher than
// this threshold, re-flooding a subtree only once it ages past it. Half the window
// gives ~2x headroom over the one-level-per-frame refresh wave. Keep strictly below
// STREAMING_HITLOD_DEMAND_WINDOW.
#define STREAMING_HITLOD_DEMAND_REFRESH (STREAMING_HITLOD_DEMAND_WINDOW / 2)

// stream_hit_demand_tree.comp.glsl coarse->fine demand descent:
// stops descending from low-detail seeds. A seed within STREAMING_HITLOD_DEMAND_PRUNE_COARSE
// levels of the coarsest (highest lod-level) still relaxes its immediate finer children's
// demand, but does not enqueue them, so the descent stops there instead of flooding the
// whole fine subtree - distant low-detail geometry does not need fine detail streamed in.
// The value counts how many coarse levels to prune (1 == coarsest level only).
// stream_hit_propagation.comp.glsl tags such a seed via the entry's top bit.
#define STREAMING_HITLOD_DEMAND_PRUNE_COARSE 0

// demand-queue entry (uint64): groupIndex in the low 32 bits, geometryID in the high
// 32. The top bit of the geometryID field carries the coarse-seed tag above.
#define STREAMING_HITLOD_DEMAND_COARSE_CHILDREN_BIT 0x80000000u  // within uint(entry >> 32)
#define STREAMING_HITLOD_DEMAND_GEOMETRY_MASK 0x7FFFFFFFu

// per-frame relaxation of the smoothed demand width (`groupHitWidths`), in log2
// encoding steps. Hits pull it finer instantly; without hits it relaxes toward
// coarse by this step (64 = 1/16 octave/frame, ~one lod level per 16 frames),
// keeping the demand stable when a group's finest hits drop out for a few frames.
#define STREAMING_HITLOD_WIDTH_RELAX 64

// lower bound on the per-instance demand scale (`instanceHitDemand`, see
// `streamingRegisterInstanceDemand`). A magnifying footprint lowers the instance's
// traversal errorScale by this factor at most, bounding the extra traversal/BLAS
// cost. Each halving is ~one lod level, so 1/128 ~ 7 levels of headroom.
#define STREAMING_HITLOD_INSTANCE_SCALE_MIN (1.0f / 128.0f)

// `instance_hit_scale` lets the filtered per-instance scale step back coarser only
// after this many frames without a demand at its current level
#define STREAMING_HITLOD_INSTANCE_HOLD_FRAMES 32

/////////////////////////////////////////

struct StreamingRequest
{
  uint maxLoads;
  uint maxUnloads;
  uint loadCounter;
  uint unloadCounter;
#ifdef __cplusplus
  union
  {
    uint64_t frameIndex;
    uint32_t frameIndexU32[2];
  };
#else
  uint64_t frameIndex;  // already embeds STREAMING_INVALID_ADDRESS_START
#endif

  // for compaction based clas management
  uint64_t clasCompactionUsedSize;
  uint     clasCompactionCount;
  // for persistent allocator clas management
  uint     clasAllocatedMaxSizedLeft;
  uint64_t clasAllocatedUsedSize;
  uint64_t clasAllocatedWastedSize;

  uint taskIndex;
  uint errorUpdate;
  uint errorAgeFilter;
  uint errorHitPropagation;
  // context of errorHitPropagation, see `stream_hit_propagation.comp.glsl`
  uint errorHitGeometryID;
  uint errorHitGroupID;
  uint errorHitIndex;  // code 2: generating group index, code 3: finer group's residentID
  uint errorHitLimit;  // code 2: geometry's group count, code 3: maxResidentGroups
  uint errorClasNotFound;
  uint errorClasList;
  uint errorClasAlloc;
  uint errorClasDealloc;
  int  errorClasUsedVsAlloc;

  BUFFER_REF(uvec2s_inout) loadGeometryGroups;
  BUFFER_REF(uvec2s_inout) unloadGeometryGroups;
};

/////////////////////////////////////////

struct ClasBuildInfo
{
  uint32_t clusterID;
  uint32_t clusterFlags;

#define ClasBuildInfo_packed_triangleCount 0 : 9
#define ClasBuildInfo_packed_vertexCount 9 : 9
#define ClasBuildInfo_packed_positionTruncateBitCount 18 : 6
#define ClasBuildInfo_packed_indexType 24 : 4
#define ClasBuildInfo_packed_opacityMicromapIndexType 28 : 4
  uint32_t packed;

  // struct VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV
  // {
  //   uint32_t geometryIndex : 24;
  //   uint32_t reserved : 5;
  //   uint32_t geometryFlags : 3;
  // };
#define ClasGeometryFlag_CULL_DISABLE_BIT_NV (1 << 29)
#define ClasGeometryFlag_NO_DUPLICATE_ANY_HIT_BIT_NV (2 << 29)
#define ClasGeometryFlag_OPAQUE_BIT_NV (4 << 29)
  uint32_t baseGeometryIndexAndFlags;

  uint16_t indexBufferStride;
  uint16_t vertexBufferStride;
  uint16_t geometryIndexAndFlagsBufferStride;
  uint16_t opacityMicromapIndexBufferStride;
  uint64_t indexBuffer;
  uint64_t vertexBuffer;
  uint64_t geometryIndexAndFlagsBuffer;
  uint64_t opacityMicromapArray;
  uint64_t opacityMicromapIndexBuffer;
};
BUFFER_REF_DECLARE_ARRAY(ClasBuildInfos_inout, ClasBuildInfo, , 16);

struct StreamingPatch
{
  // unload or loading
  uint32_t geometryID;
  uint32_t groupID;
  uint64_t groupAddress;

  // loading
  uint16_t clusterCount;
  uint16_t lodLevel;
  uint32_t groupResidentID;
  uint32_t clusterResidentID;
  uint32_t clasBuildOffset;
};
BUFFER_REF_DECLARE_ARRAY(StreamingPatchs_in, StreamingPatch, , 16);

struct StreamingGeometryPatch
{
  uint32_t geometryID;
  uint16_t discreteLodLevel;  // see Geometry::discreteLodLevel
  uint16_t cachedBlasClustersCount;
  uint64_t cachedBlasAddress;
};
BUFFER_REF_DECLARE_ARRAY(StreamingGeometryPatchs_in, StreamingGeometryPatch, , 16);

struct StreamingUpdate
{
  // unload operations are before load operations
  uint patchUnloadGroupsCount;
  // total operations
  uint patchGroupsCount;

  // geometry patch count
  uint patchCachedBlasCount;
  uint patchCachedClustersCount;

  // all newly loaded groups have linear positions in the
  // compacted list of active groups starting with this value
  uint loadActiveGroupsOffset;
  uint loadActiveClustersOffset;

  uint taskIndex;
  uint frameIndex;

  // loaded come first, then unloaded
  BUFFER_REF(StreamingPatchs_in) patches;

  // ray tracing: per-frame CLAS-build vertex-positions.
  // Positions are streamed here instead of being kept in the persistent group blob.
  uint64_t clasVerticesBuffer;
  uint32_t clasVerticesGroupStride;  // bytes reserved per group slot (max verts/group * sizeof(vec3))

  // newly loaded group clusters fill these
  BUFFER_REF(ClasBuildInfos_inout) newClasBuilds;
  BUFFER_REF(uint32s_inout) newClasResidentIDs;
  BUFFER_REF(uint32s_inout) newClasSizes;
  BUFFER_REF(uint64s_inout) newClasAddresses;
  uint32_t newClasCount;

  uint32_t newClasGeometryIndicesTaskCounter;
  BUFFER_REF(uint32s_inout) newClasGeometryIndices;

  DispatchIndirectCommand dispatchClasGeometryIndices;


  // geometry state handling
  BUFFER_REF(StreamingGeometryPatchs_in) geometryPatches;

  // compaction
  uint32_t moveClasCounter;
  uint64_t moveClasSize;
  BUFFER_REF(uint64s_inout) moveClasSrcAddresses;
  BUFFER_REF(uint64s_inout) moveClasDstAddresses;
};

/////////////////////////////////////////

struct StreamingGroup
{
  uint32_t geometryID;
  uint16_t lodLevel;
  uint16_t age;
  BUFFER_REF(Group_in) group;
};
BUFFER_REF_DECLARE_ARRAY(StreamingGroup_inout, StreamingGroup, , 16);

struct StreamingResident
{
  // Dynamic content:
  //
  // These are lists of groups that don't include
  // lowest detail, as those are always kept resident

  uint activeGroupsCount;
  uint activeClustersCount;

  uint loGroupsCount;
  uint loClustersCount;

  BUFFER_REF(uint32s_in) activeGroups;

  // Resident content:
  //
  // Due to immutability of the residentID, these tables
  // are indexed sparsely, and they do contain all
  // content, including persistent lowest detail.

  BUFFER_REF(StreamingGroup_inout) groups;
  BUFFER_REF(uint32s_inout) groupIDs;

  // only if persistent clas allocator is used
  BUFFER_REF(uvec2s_inout) groupClasSizes;

  BUFFER_REF(uint64s_inout) clusters;
  BUFFER_REF(uint64s_inout) clasAddresses;
  BUFFER_REF(uint32s_inout) clasSizes;

  // hit tracking
  // packed per-group hit state: cyclic frame of the last hit (own or inherited
  // from finer groups), the smallest recent cone width, and whether no finer
  // group is resident. See the STREAMING_HITFRAME_* defines.
  BUFFER_REF(uint32s_inout) groupHitFrames;
  // smoothed demand width per group, same inverted encoding as the hit width bits
  // (larger = finer, zero = no demand). Maintained by stream_hit_propagation.
  BUFFER_REF(uint32s_inout) groupHitWidths;
  // resident cluster index -> resident group index
  BUFFER_REF(uint32s_in) clusterGroups;

  uint64_t clasBaseAddress;
  uint64_t clasMaxSize;

  // single element address for persistent information storage
  // we read from these values when no load/unloads are performed per frame
  //
  // for compaction based clas management
  BUFFER_REF(uint64s_inout) clasCompactionUsedSize;
  // for persistent allocator clas management
  BUFFER_REF(uint32s_inout) clasAllocatedMaxSizedLeft;

  // aid debugging
  uint taskIndex;
  uint frameIndex;
};

/////////////////////////////////////////

struct AllocatorRange
{
  int32_t  count;
  uint32_t offset;
};
BUFFER_REF_DECLARE_ARRAY(AllocatorRange_inout, AllocatorRange, , 8);
struct AllocatorStats
{
  int64_t allocatedSize;
  int64_t wastedSize;
};
BUFFER_REF_DECLARE(AllocatorStats_inout, AllocatorStats, , 8);


struct StreamingAllocator
{
  uint freeGapsCounter;
  uint granularityByteShift;  // size of one unit in (1 << shift) bytes
  uint maxAllocationSize;     // not in bytes but units
  uint sectorCount;
  uint sectorMaxAllocationSized;
  uint sectorSizeShift;
  uint baseWastedSize;
  uint usedBitsCount;

  DispatchIndirectCommand dispatchFreeGapsInsert;

  BUFFER_REF(uint32s_inout) freeGapsPos;
  BUFFER_REF(uint16s_inout) freeGapsSize;
  BUFFER_REF(uint32s_inout) freeGapsPosBinned;
  BUFFER_REF(AllocatorRange_inout) freeSizeRanges;
  BUFFER_REF(uint32s_inout) usedBits;
  BUFFER_REF(uint32s_inout) usedSectorBits;
  BUFFER_REF(AllocatorStats_inout) stats;
};

/////////////////////////////////////////

struct SceneStreaming
{
  int32_t ageThreshold;
  uint    frameIndex;
  uint    useBlasCaching;
  uint    clasPositionTruncateBits;
  uint    useHitStreaming;
  uint    maxResidentGroups;
  uint    useHitStreamingLod;
  // scales the acceptable error derived from hit cone widths (default 1.0,
  // higher values clamp to coarser detail)
  float hitLodWidthFactor;

  // see GEOMETRY_CACHED_BLAS_STRIDE. 0 means no entry.
  BUFFER_REF(uint64s_inout) cachedBlasAddresses;

  StreamingResident  resident;
  StreamingUpdate    update;
  StreamingRequest   request;
  StreamingAllocator clasAllocator;
};


/////////////////////////////////////////

// Visualization of the persistent clas allocator's memory state.
// see `StreamingAllocatorVis` and `stream_allocator_vis.comp.glsl`

BUFFER_REF_DECLARE(SceneStreaming_in, SceneStreaming, readonly, 8);

#define STREAM_ALLOCATOR_VIS_PASS_MEMORY 0
#define STREAM_ALLOCATOR_VIS_PASS_GROUPS 1
#define STREAM_ALLOCATOR_VIS_PASS_HISTOGRAM 2

// allocation sizes run from 1 to `maxAllocationSize` units, binned down to this on the device
#define STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS 256

struct StreamingAllocatorVisConstants
{
  uint64_t streamingAddress;

  // sub-rectangle of the image that maps to allocator memory, sized to the pixels the UI
  // has for it. `usedHeight` counts image rows, so it is `rowCount * rowPitch`.
  uint usedWidth;
  uint usedHeight;
  // image rows per memory row, the ones past `rowContent` are left blank as a separator
  uint rowPitch;
  uint rowContent;
  // a row is a whole number of sectors, so an allocation never straddles one
  uint unitsPerPixel;
  uint unitsPerRow;
  uint totalUnits;

  uint groupCount;
  uint colorXor;

  // histogram pass, `STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS` counters
  uint64_t histogramAddress;
  // in units, the range the histogram covers
  uint maxAllocationSize;
};

#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_STREAMING_H_