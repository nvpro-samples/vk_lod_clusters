/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio_streaming.h"
#include "shaderio_ptlas.h"

#ifndef _SHADERIO_BUILDING_H_
#define _SHADERIO_BUILDING_H_

#define TRAVERSAL_INVALID_LOD_LEVEL 0xFF

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else

#define INSTANCE_VISIBLE_BIT 1
#define INSTANCE_USES_MERGED_BIT 2
#define INSTANCE_USES_DISCRETE_BIT 4
// tags a traversal item that was seeded at a discrete lod level. Its nodes are
// always descended and all clusters of its groups are rendered.
// Stored in `TraversalInfo::instanceID`, which leaves 30 bits for the instance.
#define TRAVERSAL_DISCRETE_BIT 0x80000000u
// tags a traversal item whose lod metric sphere failed last frame's hiz,
// see `USE_CULLED_GROUP_LOD`
#define TRAVERSAL_OCCLUDED_BIT 0x40000000u
#ifndef USE_CULLED_GROUP_LOD
#define USE_CULLED_GROUP_LOD 0
#endif
#define INSTANCE_USES_LOWDETAIL_BIT 8
// hybrid path: the instance is traversed for rasterization only, its blas is a reused one
#define INSTANCE_HYBRID_RASTER_ONLY_BIT 16
// hybrid path: in the frustum but occluded in last frame's hiz, rasterized in the second pass
#define INSTANCE_HYBRID_OCCLUDED_BIT 32
// hybrid path: keeps a cached blas within `hybridCachedTolerance` levels, a known mismatch
#define INSTANCE_HYBRID_TOLERANCE_BIT 64

// one reject bit per cluster of a group, `GROUP_CLUSTER_COUNT` comes from the renderer
#ifndef GROUP_CLUSTER_COUNT
#define GROUP_CLUSTER_COUNT 32
#endif
#define REJECT_CLUSTER_MASK_WORDS ((GROUP_CLUSTER_COUNT + 31) / 32)

// `GeometryBuildInfo::flags`, only maintained for the "blas reuse" visualization
#define GEOMETRY_HAS_SHARING_INSTANCES 1

#define BLAS_CANDIDATE_HAS_DUPLICATES 1

#define BLAS_BUILD_INDEX_LOWDETAIL (uint(~0))
#define BLAS_BUILD_INDEX_SHARE_BIT (uint(1 << 31))
#define BLAS_BUILD_INDEX_CACHE_BIT (uint(1 << 30))
// with BLAS_BUILD_INDEX_CACHE_BIT the upper bits carry the cached blas slot,
// the lower bits the geometryID
#define BLAS_BUILD_INDEX_CACHE_SLOT_SHIFT 26
#define BLAS_BUILD_INDEX_CACHE_SLOT_MASK (uint(0xF) << BLAS_BUILD_INDEX_CACHE_SLOT_SHIFT)
#define BLAS_BUILD_INDEX_GEOMETRY_MASK (uint((1 << BLAS_BUILD_INDEX_CACHE_SLOT_SHIFT) - 1))

// disabling is more for testing
#define TRAVERSAL_ALLOW_LOW_DETAIL_BLAS true

#endif

// The item descriptor used in the lod hierarchy traversal
// producer/consumer queue.
// It can can encode a lod hierarchy node, or a cluster group of an instance.
// must fit in 64-bit
struct TraversalInfo
{
  uint32_t instanceID;
  uint32_t packedNode;
};

// A renderable cluster
// must fit in 64-bit, and can be overlayed with `TraversalInfo`
// thereore instanceID must come first.
struct ClusterInfo
{
  uint32_t instanceID;
  uint32_t clusterID;
};
BUFFER_REF_DECLARE_ARRAY(ClusterInfos_inout, ClusterInfo, , 8);

// Indirect build information to build a BLAS from an array of CLAS references
struct BlasBuildInfo
{
  // the number of CLAS that this BLAS references
  uint32_t clusterReferencesCount;
  // stride of array (typically 8 for 64-bit)
  uint32_t clusterReferencesStride;
  // start address of the array
  uint64_t clusterReferences;
};
BUFFER_REF_DECLARE_ARRAY(BlasBuildInfo_inout, BlasBuildInfo, , 16);

// Only used for USE_BLAS_DEDUP
//
// A BLAS candidate is what an instance would build on its own. When multiple candidates
// reference the same set of CLAS, only the representative is built.
// Unlike `BlasBuildInfo` this struct isn't read by the driver, so the dedup key and
// result can live next to the reference list.
struct BlasCandidateInfo
{
  // start address of the CLAS reference array, sub-allocated from `blasClusterAddresses`
  uint64_t clusterReferences;
  // the number of CLAS that this BLAS references, also third component of the key
  uint32_t clusterReferencesCount;
  // order-invariant accumulation of the per-CLAS hashes
  uint32_t keyXor;
  uint32_t keyAdd;
  // candidate index of the unique BLAS to use, itself when unique
  uint32_t representative;
  // index into the build arrays, only written by representatives
  uint32_t buildIndex;
  // BLAS_CANDIDATE_HAS_DUPLICATES if others resolved to this candidate
  uint32_t flags;
};
BUFFER_REF_DECLARE_ARRAY(BlasCandidateInfos_inout, BlasCandidateInfo, , 32);

// Indirect build information for a TLAS instance
struct TlasInstance
{
  mat3x4   worldMatrix;
  uint32_t instanceCustomIndex24_mask8;
  uint32_t instanceShaderBindingTableRecordOffset24_flags8;
  uint64_t blasReference;
};
BUFFER_REF_DECLARE_ARRAY(TlasInstances_inout, TlasInstance, , 16);

struct GeometryBuildHistogram
{
  // number of instances that start at this lod-level
  uint32_t lodLevelMinHistogram[SHADERIO_MAX_LOD_LEVELS];
  // number of instances that end at this lod-level
  uint32_t lodLevelMaxHistogram[SHADERIO_MAX_LOD_LEVELS];
  // a "somewhat" stable instance to use for sharing, that ends
  // at this lod-level
  uint32_t lodLevelMaxPackedInstance[SHADERIO_MAX_LOD_LEVELS];
};
BUFFER_REF_DECLARE_ARRAY(GeometryBuildHistogram_inout, GeometryBuildHistogram, , 16);

// blas sharing / merging state, padded so all fields come from one 16 byte load
struct GeometryBuildInfo
{
  // blas sharing
  uint8_t  shareLevelMin;    // highest potential detail
  uint8_t  shareLevelMax;    // lowest potential detail
  uint32_t shareInstanceID;  // which instance to use for sharing (its lod range is expressed by the values above)
  // blas merging
  uint32_t mergedInstanceID;  // which instance to use for triggering the merged traversal
  uint32_t flags;             // GEOMETRY_HAS_SHARING_INSTANCES, visualization only
};
BUFFER_REF_DECLARE_ARRAY(GeometryBuildInfos_inout, GeometryBuildInfo, , 16);
BUFFER_REF_DECLARE_SIZE(GeometryBuildInfo_size, GeometryBuildInfo, 16);

// blas caching state, kept apart from the sharing state as the two are independent.
// The streaming age filter reads `cachedLevel` for every resident group, so it stays dense.
struct GeometryCachedInfo
{
  uint32_t cachedLevel;       // lowest lod level of any instance using the cached blas, invalid if none
  uint32_t cachedBuildIndex;  // per-frame blas build that the cached blas is copied from
};
BUFFER_REF_DECLARE_ARRAY(GeometryCachedInfos_inout, GeometryCachedInfo, , 8);

struct InstanceBuildInfo
{
  // how many clusters this instance uses
  uint32_t clusterReferencesCount;

  // which blas to use
  // Can be BLAS_BUILD_INDEX_LOWDETAIL or have BLAS_BUILD_INDEX_SHARE_BIT set.
  // With BLAS_BUILD_INDEX_SHARE_BIT it encodes the instanceID that it shares the blas from.
  // Without any of the previous special cases it is the index into the array of the
  // dynamically built BLAS.
  uint32_t blasBuildIndex;

  uint8_t  lodLevelMin;          // highest potential detail
  uint8_t  lodLevelMax;          // lowest potential detail
  uint16_t geometryLodLevelMax;  // the highest lod level of the geometry
  uint32_t geometryID;
};
BUFFER_REF_DECLARE_ARRAY(InstanceBuildInfos_inout, InstanceBuildInfo, , 16);

// The central structure that contains relevant information to
// perform the runtime lod hierchy traversal and building of
// all relevant clusters to be rendered in the current frame.
// (not optimally packed for cache efficiency but readability)
struct SceneBuilding
{
  mat4 traversalViewMatrix;
  mat4 cullViewProjMatrix;
  mat4 cullViewProjMatrixLast;

  // for two pass culling
  uint cullPass;
  uint frameIndex;

  // for USE_TWO_PASS_REJECT_LISTS
  uint rejectInstanceCounter;
  uint rejectNodeCounter;
  uint rejectGroupCounter;
  // number of groups that pass 0 traversed, `rejectClusterMasks` is indexed by the same slot
  uint pass0GroupCount;

  uint numGeometries;
  uint numRenderInstances;
  uint maxRenderClusters;
  uint maxTraversalInfos;

  float errorOverDistanceThreshold;
  float culledErrorScale;
  // USE_HIT_STREAMING_LOD: a group also opens while the camera metric at this multiple
  // of the errorScale wants it, regardless of ray demand. 0 disables.
  float hitCameraErrorScale;
  float swRasterThreshold;

  uint sharingEnabledLevels;
  uint sharingTolerantLevels;
  uint sharingPushCulled;

  // extra lod levels an instance may give up to reach the cached blas
  uint cachingPushLevels;
  // additional levels granted to an instance already on the cached blas, keeps it there
  uint cachingStickyLevels;
  // hybrid path: a visible instance keeps a cached blas within this many levels of its cut, -1 off
  int hybridCachedTolerance;

  // Retention value the streaming age filter reads, one per geometry: the minimum of
  // `cachedLevel` over a revalidation cycle, not of a single frame, so a skipped partition
  // cannot have the groups its instances still reference aged out. Separate from
  // `GeometryCachedInfo`, which is re-filled with ~0 every frame for the atomicMin.
  BUFFER_REF(uint32s_inout) geometryCachedRetain;
  // the cycle currently accumulating; it only becomes the live value above once complete,
  // because the reset frame and a partition's forced revalidation are on different clocks
  BUFFER_REF(uint32s_inout) geometryCachedRetainNext;
  // 1 on the frame the cycle restarts, then `retainNext` goes live and a new one starts
  uint cachedRetainReset;

  // force a stable partition to be reprocessed after this many frames, 0 never
  uint partitionRevalidateFrames;
  // process partitions crossing or entering the frustum every frame
  uint partitionFrustumRevalidate;
  // only those whose nearest corner is closer than this view depth
  float partitionFrustumDepth;
  // invalidate everything this frame, set when streaming moved a cached blas
  uint partitionInvalidateAll;
  uint numPartitions;
  // counters, reset per frame
  uint partitionActiveCounter;

  // discrete lod (rasterization)
  // number of tail lod levels that may use a single discrete level
  uint discreteEnabledLevels;
  // maximum lod levels an instance's lod range may span, 0 disables the test
  uint discreteLodRange;

  uint renderClusterCounter;
  uint renderClusterCounterSW;
  uint renderClusterCounterAlpha;
  uint renderClusterCounterAlphaSW;

  uint traversalNodeReadCounter;
  uint traversalNodeWriteCounter;
  uint traversalGroupWriteCounter;

  // persistent traversal kernel
  int traversalTaskCounter;

  // multi-pass traversal kernel
  uint traversalPass;
  uint traversalNodeStart;
  uint traversalNodeEnd;
  uint traversalGroupStart;
  uint traversalGroupEnd;

  // faster hit-streaming demand BFS (post-traversal, USE_HIT_STREAMING). It reuses
  // the `traversalNodeInfos` storage as its work queue (free once traversal is done)
  // but keeps its own counters so it never interferes with the traversal state.
  // Each queue entry packs {geometryID<<32 | groupIndex}; the front is descended one
  // finer lod transition per pass. See stream_hit_demand_tree.comp.glsl.
  uint hitDemandWriteCounter;  // atomic append cursor into traversalNodeInfos
  uint hitDemandStart;         // current frontier pass processes [start,end)
  uint hitDemandEnd;
  uint hitDemandPass;

  // result of traversal init & scratch for traversal run
  // array size is [maxTraversalInfos]
  // `setupSecondPass` swaps this to `rejectNodeInfos` for the second cull pass
  BUFFER_REF(uint64s_coh_volatile) traversalNodeInfos;
  // array size is [maxTraversalInfos]
  // `setupSecondPass` swaps this to `rejectGroupInfos` for the second cull pass
  BUFFER_REF(uint64s_coh_volatile) traversalGroupInfos;

  // for USE_TWO_PASS_REJECT_LISTS, all filled during the first cull pass.
  //
  // The second pass consumes them instead of traversing the hierarchy again:
  // `traversalNodeInfos`/`traversalGroupInfos` are pointed at the reject arrays
  // and seeded with their counters, so the traversal kernels stay unchanged.

  // instances that failed the first pass, array size is [numRenderInstances]
  BUFFER_REF(uint32s_inout) rejectInstances;
  // inner nodes that failed the first pass, array size is [maxTraversalInfos]
  BUFFER_REF(uint64s_coh_volatile) rejectNodeInfos;
  // group leaves that failed the first pass, array size is [maxTraversalInfos]
  BUFFER_REF(uint64s_coh_volatile) rejectGroupInfos;
  // per cluster reject bits for each group the first pass traversed,
  // array size is [maxTraversalInfos * REJECT_CLUSTER_MASK_WORDS]
  BUFFER_REF(uint32s_inout) rejectClusterMasks;
  // stays on the first pass array while `traversalGroupInfos` is swapped away
  BUFFER_REF(uint64s_inout) pass0GroupInfos;

  // result of traversal run or separate_groups
  // array size is [maxRenderClusters]
  BUFFER_REF(ClusterInfos_inout) renderClusterInfos;

  // only for rasterization so we have two lists
  BUFFER_REF(ClusterInfos_inout) renderClusterInfosAlpha;

  // the per-instance kernels dispatch over the active partitions' instances.
  // All three use a workgroup of 64, so one command serves them.
  DispatchIndirectCommand indirectDispatchActiveInstances;
  DispatchIndirectCommand indirectDispatchNodes;
  DispatchIndirectCommand indirectDispatchGroups;
  DispatchIndirectCommand indirectDispatchRejectInstances;
  DispatchIndirectCommand indirectDispatchRejectClusters;
  DispatchIndirectCommand indirectDispatchHitDemand;

  // rasterization related
  //////////////////////////////////////////////////

  // hw rasterization via nv
  DrawMeshTasksIndirectCommandNV indirectDrawClustersNV;
  DrawMeshTasksIndirectCommandNV indirectDrawClustersAlphaNV;
  DrawMeshTasksIndirectCommandNV indirectDrawClusterBoxesNV;
  DrawMeshTasksIndirectCommandNV indirectDrawClusterBoxesAlphaNV;

  // hw rasterization via ext
  DrawMeshTasksIndirectCommandEXT indirectDrawClustersEXT;
  DrawMeshTasksIndirectCommandEXT indirectDrawClustersAlphaEXT;
  DrawMeshTasksIndirectCommandEXT indirectDrawClusterBoxesEXT;
  DrawMeshTasksIndirectCommandEXT indirectDrawClusterBoxesAlphaEXT;
  uint                            numRenderedClusters;
  uint                            numRenderedClustersAlpha;

  // sw rasterization
  DispatchIndirectCommand indirectDispatchClustersSW;
  DispatchIndirectCommand indirectDispatchClustersAlphaSW;
  uint                    numRenderedClustersSW;
  uint                    numRenderedClustersAlphaSW;

  // clusters classified for sw-rasterization.
  // We simply have two lists, we could use a single list
  // and append front or back to conserve memory, however
  // for simplicity two are used.
  BUFFER_REF(ClusterInfos_inout) renderClusterInfosSW;
  BUFFER_REF(ClusterInfos_inout) renderClusterInfosAlphaSW;

  // ray tracing related
  //////////////////////////////////////////////////

  // the scene's acceleration structure, a tlas or a ptlas. Always referenced by
  // address, a ptlas has no acceleration structure object to bind.
  uint64_t tlasAddress;

  DispatchIndirectCommand indirectDispatchBlasInsertion;

  // hybrid path (USE_HYBRIDPATH): the clusters rasterized for primary visibility.
  // A primary visible instance's own blas is built from the same cut.
  // Array size is [maxRenderClusters]
  BUFFER_REF(ClusterInfos_inout) hybridClusterInfos;
  uint                            hybridClusterCounter;
  uint                            numHybridClusters;
  DrawMeshTasksIndirectCommandNV  indirectDrawHybridNV;
  DrawMeshTasksIndirectCommandEXT indirectDrawHybridEXT;
  // the clusters occluded in last frame's hiz, the second raster pass draws those passing
  // the first pass' hiz. Array size is [maxRenderClusters]
  BUFFER_REF(ClusterInfos_inout) hybridRejectClusterInfos;
  uint                    hybridRejectCounter;
  uint                    numHybridRejectClusters;
  DispatchIndirectCommand indirectDispatchHybridReject;

  // Computed dynamically, total number of CLAS that are referenced
  // across all BLAS built in a frame.
  uint blasClasCounter;

  // Computed dynamically and contains the number of blas that are built in a frame.
  // It is read indirectly by `vkCmdBuildClusterAccelerationStructureIndirectNV`
  uint blasBuildCounter;

  // per scene instance
  // ------------

  // culling/visibility related information
  BUFFER_REF(uint8s_inout) instanceVisibility;
  // 1 if the instance used its geometry's cached blas last frame, persists across frames
  BUFFER_REF(uint8s_inout) instanceCachedSticky;

  // Only for USE_HIT_STREAMING_LOD. Per-instance demand for detail *finer* than
  // the camera-distance metric would pick (e.g. a magnifying glass / converging
  // refraction). Stores the absolute `errorScale` a ray footprint demanded (< 1
  // is finer), or a large FLT_MAX "no demand" sentinel. Combined with min() into
  // the camera/culled errorScale everywhere a metric is evaluated, so only
  // demanding instances refine (overriding even culledErrorScale for a
  // mirror-only-visible instance) and the blas-sharing classification learns they
  // need their own finer blas. Written by `streamingRegisterInstanceDemand`
  // (atomicMin over the frame's ray hits), consumed and reset by
  // `instance_hit_scale`. See the design in docs/hitbased_streaming.md.
  BUFFER_REF(uint32s_inout) instanceHitDemand;
  // filtered `instanceHitDemand`, what classify and traversal actually read:
  // bits 0..7 the level, errorScale = 2^-level with 0 = no demand, bits 8..31 the
  // frames the level has been held without demand.
  BUFFER_REF(uint32s_inout) instanceHitScale;
  // per geometry, the finest lod level any traversal can open next frame given the ray
  // demand, classify raises `lodLevelMin` to it. Written by `stream_hit_propagation`,
  // ~0 when it did not run.
  BUFFER_REF(uint32s_inout) geometryHitDemandLevel;

  // instance sorting
  BUFFER_REF(uint32s_inout) instanceSortValues;
  BUFFER_REF(uint32s_inout) instanceSortKeys;

  // instance blas handling
  BUFFER_REF(InstanceBuildInfos_inout) instanceBuildInfos;
  BUFFER_REF(TlasInstances_inout) tlasInstances;

  // per blas build
  // --------------
  // (arrays are sized for worst-case to be per-instance)

  // configured in `blas_setup_insertion.comp.glsl`
  // and input to `vkCmdBuildClusterAccelerationStructureIndirectNV`
  BUFFER_REF(BlasBuildInfo_inout) blasBuildInfos;

  // result of building operations
  BUFFER_REF(uint32s_inout) blasBuildSizes;
  BUFFER_REF(uint64s_inout) blasBuildAddresses;

  // split into per-blas regions during
  // `blas_setup_insertion.comp.glsl`
  // array size is [maxRenderClusters]
  BUFFER_REF(uint64s_inout) blasClusterAddresses;

  // for USE_BLAS_DEDUP
  // ------------------

  // configured in `blas_setup_insertion.comp.glsl`
  // and resolved in `blas_dedup_scan.comp.glsl`
  BUFFER_REF(BlasCandidateInfos_inout) blasCandidateInfos;
  // candidate index per `blasClusterAddresses` entry, written during insertion
  BUFFER_REF(uint32s_inout) blasCandidateSlots;

  // finds equivalent candidates, `blasDedupTableMask + 1` entries.
  // stores `candidateIndex + 1`, so zero means empty and a fill clears it.
  BUFFER_REF(uint32s_inout) blasDedupTable;
  uint32_t blasDedupTableMask;

  // Computed dynamically, number of BLAS candidates in a frame.
  // `blasBuildCounter` is how many of those are unique.
  uint32_t blasCandidateCounter;

  // per scene geometry
  // ------------
  BUFFER_REF(GeometryBuildInfos_inout) geometryBuildInfos;
  BUFFER_REF(GeometryCachedInfos_inout) geometryCachedInfos;
  BUFFER_REF(GeometryBuildHistogram_inout) geometryHistograms;

  // for USE_PTLAS
  // ------------
  // number of instances appended to `ptlasInstanceUpdates` this frame
  uint ptlasUpdateCounter;
  // blas changes of instances that were written before
  BUFFER_REF(PtlasInstanceUpdates_inout) ptlasInstanceUpdates;
  // last blas reference written to the ptlas, per instance
  BUFFER_REF(uint64s_inout) ptlasInstanceBlas;
  // number of instances updated within a partition, per frame statistic
  BUFFER_REF(uint32s_inout) ptlasPartitionUpdates;

  // for USE_PARTITION_EARLY_OUT
  // ------------
  // Per-partition early-out. Membership is fixed, so the instance order and the
  // per-partition ranges are built once on the host. None of this needs the ptlas: the
  // tlas path skips the same per-instance kernels, it just cannot skip its own build.
  // partition of each instance, indexed by instance id
  BUFFER_REF(uint32s_in) instancePartitions;
  // instance ids grouped by partition
  BUFFER_REF(uint32s_in) partitionInstanceOrder;
  // x = first entry in `partitionInstanceOrder`, y = instance count.
  // Also allocated for USE_PTLAS, its statistics read the instance count.
  BUFFER_REF(uvec2s_in) partitionRanges;
  // instances in the partition that did NOT resolve to a persistent blas, counted by
  // `traversal_init_blas_reuse`. 0 means the partition is stable and can be skipped:
  // every instance sits on a cached or pre-built low detail blas, so there is no
  // traversal, no blas build and no ptlas update to do for it.
  BUFFER_REF(uint32s_inout) partitionNonPersistent;
  // frame index a partition was last processed, drives forced revalidation
  BUFFER_REF(uint32s_inout) partitionLastFrame;
  // world bounds of each partition
  BUFFER_REF(BBoxes_in) partitionBboxes;
  // 1 while the partition was inside the frustum last frame, catches camera cuts
  BUFFER_REF(uint32s_inout) partitionVisible;
  // one bit per block of PARTITION_INSTANCE_BLOCK_SIZE partition-sorted instances, set by
  // the partitions that must be processed this frame
  BUFFER_REF(uint32s_inout) partitionBlockBits;
  // compacted block ids, each one a start offset into `partitionInstanceOrder`
  BUFFER_REF(uint32s_inout) activeBlocks;
  uint numInstanceBlocks;
  uint activeBlockCounter;
  // the operation of this frame's build, and the number of operations
  BUFFER_REF(PtlasOperations_inout) ptlasOperations;
  BUFFER_REF(uint32s_inout) ptlasOperationCount;

  // for USE_BLAS_CACHING
  BUFFER_REF(uint64s_inout) cachedBlasClusterAddressesSrc;
  BUFFER_REF(uint64s_inout) cachedBlasClusterAddressesDst;

  uint32_t cachedBlasCopyCounter;
};


#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_BUILDING_H_