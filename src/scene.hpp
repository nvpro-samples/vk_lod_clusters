/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>
#include <array>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <functional>
#include <regex>
#include <span>

#include <glm/glm.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/alignment.hpp>

#include "serialization.hpp"
#include "scene_kdop.hpp"
#include "meshopt_clusterlod.h"
#include "../shaders/shaderio_scene.h"

namespace lodclusters {

// current process memory usage, all in bytes, zero if unsupported.
// `privateCommit` excludes memory mapped file pages, unlike `workingSet`.
struct ProcessMemoryUsage
{
  uint64_t workingSet        = 0;
  uint64_t peakWorkingSet    = 0;
  uint64_t privateCommit     = 0;
  uint64_t peakPrivateCommit = 0;
};

ProcessMemoryUsage getProcessMemoryUsage();

// Controls the scene's data generation during loading and processing.
struct SceneConfig
{
  static const uint32_t version = 3;

  // cluster and cluster group settings
  uint32_t clusterVertices    = 128;
  uint32_t clusterTriangles   = 128;
  uint32_t clusterGroupSize   = 32;
  uint32_t preferredNodeWidth = 8;

  // default setting should prefer ray tracing
  bool meshoptPreferRayTracing = true;

  // store groups in a compressed way
  // uncompress at runtime
  bool useCompressedData = true;

  // allow materials
  bool enableMultiMaterials = false;
  bool _reserved            = false;

  // due to the simple shading, only enable normals for now
  uint32_t enabledAttributes = shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;

  // settings that affect clusterization
  float meshoptFillWeight  = 0.5f;  // if ray-tracing is preferred
  float meshoptSplitFactor = 2.0f;  // otherwise

  // at each lod step reduce cluster group triangles by this factor
  float lodLevelDecimationFactor = 0.5f;

  // lod error propagation for meshoptimizer's clusterlod
  // These control the error propagation across lod levels to
  // account for simplifying an already simplified mesh.
  // error = max(previousError * lodErrorMergePrevious, currentError) +
  //         lodErrorMergeAdditive * currentError;
  float lodErrorMergePrevious = 1.5;
  float lodErrorMergeAdditive = 0.0f;

  // mesh simplification weights for attributes
  // zero to disable
  float simplifyNormalWeight      = 0.5f;
  float simplifyTangentWeight     = 0.0f;
  float simplifyTangentSignWeight = 0.2f;
  float simplifyTexCoordWeight    = 0.5f;
  float simplifyMaterialWeight    = 0.1f;

  // used when compression is enabled
  uint32_t compressionPosDropBits = 7;
  uint32_t compressionTexDropBits = 7;

  // experimental meshoptimizer, try to remove small triangles despite high error
  float lodErrorEdgeLimit = 1.0;

  // clamp the attribute error to the position error scale, avoids overly conservative lod picking
  bool simplifyErrorClamped = true;
  // try to keep fold lines between opposite-facing triangles, at a small processing cost
  bool simplifyPreserveFolds = false;
  // dilate open cluster borders to compensate area loss, mostly useful for foliage.
  // `All` as in every geometry, the two-sided variant below is the selective one.
  bool simplifyDilateBordersAll = false;
  // dilate the borders of geometries that use a two-sided material, which is what foliage
  // typically is. Only ever adds dilation, it does not turn off `simplifyDilateBordersAll`.
  bool simplifyDilateBordersTwoSided = true;

  // triangle order optimization within a cluster, see `meshopt_optimizeMeshletLevel`
  uint32_t optimizeClustersLevel = 1;

  // want to allow some binary compatibility with older cache files
  // safe to add new variables into this section as long as they are zeroed by default
  uint32_t reservedData[12] = {};
};

// Per-mesh overrides of the simplification settings, loaded from a json file
// (`SceneLoaderConfig::simplifyOverridesFile`). Each entry matches glTF mesh names with a
// regular expression and replaces the global `SceneConfig` values for the geometries built
// from them, so one scene can mix e.g. aggressive foliage settings with conservative ones.
//
//   [
//     { "mesh": "leaves_.*",  "simplifyuvweight": 1.0, "simplifydilateall": true },
//     { "mesh": "leaves_lod", "simplifyuvweight": 0.25 }
//   ]
//
// All matching entries are applied in file order, so a later entry wins over an earlier one.
struct SimplifyOverrides
{
  // a `SceneConfig` member that an entry may replace; `key` is the json name, which is the
  // same name the equivalent command line option uses
  struct Field
  {
    enum Kind : uint8_t
    {
      eFloat,
      eBool,
      eUint,
    };

    const char* key;
    Kind        kind;
    size_t      offset;  // into SceneConfig
  };

  static std::span<const Field> getFields();

  struct Entry
  {
    std::string pattern;
    std::regex  regex;
    // bit per `getFields()` entry, tells which members of `values` are meaningful
    uint32_t    setMask = 0;
    SceneConfig values  = {};
  };

  std::vector<Entry> entries;

  bool empty() const { return entries.empty(); }

  // parses the json file, logs and returns false on error
  bool load(const std::filesystem::path& filePath);

  // applies every entry whose regex matches `meshName`, in file order.
  // `config` may be null when only the hash is of interest.
  // returns a hash over the entries that matched, 0 when none did.
  uint64_t match(const std::string& meshName, SceneConfig* config) const;
};

// Phases reported during asynchronous scene loading, exposed through
// SceneLoadState::phase so the UI can label the progress bar.
enum class LoadPhase : uint32_t
{
  ProcessingScene,  // building LOD clusters from raw geometry
  LoadingScene,     // loading pre-processed clusters from the cache
  ProbingTextures,  // scanning texture files for the memory budget
  LoadingTextures,  // uploading textures to the GPU
};

inline const char* getLoadPhaseName(uint32_t phase)
{
  switch(LoadPhase(phase))
  {
    case LoadPhase::ProcessingScene:
      return "Processing Scene";
    case LoadPhase::LoadingScene:
      return "Loading Scene";
    case LoadPhase::ProbingTextures:
      return "Probing Textures";
    case LoadPhase::LoadingTextures:
      return "Loading Textures";
    default:
      return "Loading Scene";
  }
}

// Cross-thread signals shared between the background scene/texture loader and the
// UI. Holds pointers to atomics owned by the app; embedded in SceneLoaderConfig and
// passed to the texture loader separately. All pointers are optional (may be null).
struct SceneProgressInfo
{
  // items completed / total in the current phase. The loader only reports the raw
  // counts; the UI derives the percentage and shows "Completed: N of T".
  // What an item is depends on the phase: triangles while geometries are loaded or processed
  // (geometries if their triangle count is not known up front), images for textures. Same unit
  // as the percentage that phase logs, so ui and log agree.
  std::atomic_uint64_t* completedCount = nullptr;
  std::atomic_uint64_t* totalCount     = nullptr;
  // current LoadPhase (stored as uint32_t)
  std::atomic_uint32_t* progressPhase = nullptr;

  // start a new phase: label it and reset the completed/total counters
  void beginPhase(LoadPhase phase, uint64_t total) const
  {
    if(progressPhase)
      progressPhase->store(uint32_t(phase));
    if(totalCount)
      totalCount->store(total);
    if(completedCount)
      completedCount->store(0);
  }

  void setCompleted(uint64_t completed) const
  {
    if(completedCount)
      completedCount->store(completed);
  }
};

// Control the loading and processing procedure of the scene.
// Not the results.
struct SceneLoaderConfig
{
  // Influence the number of geometries that can be processed in parallel.
  // Percentage of threads of maximum hardware concurrency
  float processingThreadsPct = 0.5;
  // We only process the data and save a cache file, then
  // terminate the app. This allows to greatly reduce peak memory
  // consumption during processing.
  bool processingOnly = false;
  // in processing only mode we allow partial success / resuming
  bool processingAllowPartial = false;
  // upper budget in GiB for the estimated memory of all geometries processed in parallel.
  // Prevents many large geometries from being in-flight at once, which is what drives
  // peak memory. 0 is automatic (60 % of installed memory), negative is a percentage
  // of installed memory (-50 == 50 %). Always clamped to what is currently available.
  int processingMemoryGiB = 0;

  // try load from cache file if file was found
  bool autoLoadCache = true;
  // when loading from cache file, memory map it,
  // rather than loading it into system RAM.
  bool memoryMappedCache = false;

  // if a scenes geometry data exceeds this, then always do a separate preprocess pass
  // and use the cache file afterwards
  size_t forcePreprocessMiB = size_t(2) * 1024;

  SceneProgressInfo progressInfo;

  // json file with per-mesh overrides of the simplification settings, see `SimplifyOverrides`
  std::filesystem::path simplifyOverridesFile;

  // regular expression strings to discard instances by property name
  std::string skipNodeNames;
  std::string skipMaterialNames;
  std::string skipMeshNames;

  // regular expression string, every instance below a node whose name matches shares a
  // partition. A nested match starts a new one, the instances below no matching node share
  // one more. Used by `ScenePartitionConfig::LAYOUT_NODE_REGEX`.
  std::string partitionNodeNames;

  // discard instances whose materials have these properties
  bool skipAlphaMasked  = false;
  bool skipAlphaBlended = true;

  bool enableTexturedMaterials = false;
  // skip normal map textures when textured materials are enabled
  bool skipNormalMaps = false;

  bool operator==(const SceneLoaderConfig& other) const
  {
    // ignore progressInfo (runtime pointers, not configuration)
    return processingThreadsPct == other.processingThreadsPct && processingOnly == other.processingOnly
           && processingAllowPartial == other.processingAllowPartial && processingMemoryGiB == other.processingMemoryGiB
           && autoLoadCache == other.autoLoadCache && memoryMappedCache == other.memoryMappedCache
           && forcePreprocessMiB == other.forcePreprocessMiB && simplifyOverridesFile == other.simplifyOverridesFile
           && skipNodeNames == other.skipNodeNames && skipMaterialNames == other.skipMaterialNames
           && skipMeshNames == other.skipMeshNames && partitionNodeNames == other.partitionNodeNames
           && skipAlphaMasked == other.skipAlphaMasked && skipAlphaBlended == other.skipAlphaBlended
           && enableTexturedMaterials == other.enableTexturedMaterials && skipNormalMaps == other.skipNormalMaps;
  }

  bool operator!=(const SceneLoaderConfig& other) const { return !(*this == other); }
};

// To artificially instance the full scene on a grid multiple times.
// Useful for benchmarking.
struct SceneGridConfig
{
  // fraction of copies that get their own set of geometries (0..1), the others
  // reuse those sets. 1 gives every copy unique geometries, which stresses the
  // streaming system and memory consumption a lot more.
  float     uniqueGeometries = 0.0f;
  uint32_t  numCopies        = 1;
  uint32_t  gridBits         = 13;
  glm::vec3 refShift         = {1.0f, 1.0f, 1.0f};
  float     snapAngle        = 0;
  float     minScale         = 1.0f;
  float     maxScale         = 1.0f;
};

// Assignment of the instances to spatial partitions, used by the partitioned tlas and by
// the partition early-out, which needs no extension.
struct ScenePartitionConfig
{
  // How instances are grouped. The api is handed nothing but the partition count, so the
  // grouping is the only lever on the driver's top level structure, and it drives trace
  // cost, build cost and how often a partition can go quiet all at once.
  enum Layout : uint32_t
  {
    // uniform cells over xz, ignores y and ignores density. Sized by `gridCellSize` or
    // `gridCellsPerAxis`
    LAYOUT_FIXED_GRID,
    // as above, the resolution is derived from the instance count
    LAYOUT_DERIVED_GRID,
    // top-down median splits of the instance centers, uniform occupancy
    LAYOUT_DERIVED_BVH_MEDIAN,
    // top-down sah splits over the instance bounds, trades occupancy for tighter partitions
    LAYOUT_DERIVED_BVH_SAH,
    // the derived grid's initial cells, each split into quadrants while it holds more than
    // the target
    LAYOUT_DERIVED_QUADTREE,
    // one partition per node group from `SceneLoaderConfig::partitionNodeNames`, the
    // instances below no matching node share one more
    LAYOUT_NODE_REGEX,
    LAYOUT_COUNT,
  };
  uint32_t layout = LAYOUT_DERIVED_GRID;
  // fixed grid: world units per cell, takes precedence over `gridCellsPerAxis` when non zero.
  // Cells are aligned to the world origin shifted by `gridOrigin`, not to the scene bounds.
  float     gridCellSize = 0;
  glm::vec3 gridOrigin   = {0, 0, 0};
  // fixed grid: cells along each of the two axes over the scene bounds
  uint32_t gridCellsPerAxis = 16;
  // derived layouts: instances a partition should hold, the bvh and quadtree leaf size or
  // the initial grid resolution
  uint32_t derivedTargetInstances = 64;
  // derived grid: hard cap for a single partition, the resolution is increased until it fits
  uint32_t derivedMaxInstances = 1024;
  // derived layouts: refinement stops once the used partitions would exceed this, 0 only
  // respects the device limit. The ptlas build cost tracks the partition count, so this is
  // the budget that matters, not the occupancy the refinement chases.
  uint32_t derivedMaxCount = 4096;
  // device limit for the number of partitions
  uint32_t maxPartitions = 0;

  bool operator==(const ScenePartitionConfig& other) const
  {
    return layout == other.layout && gridCellSize == other.gridCellSize && gridOrigin == other.gridOrigin
           && gridCellsPerAxis == other.gridCellsPerAxis && derivedTargetInstances == other.derivedTargetInstances
           && derivedMaxInstances == other.derivedMaxInstances && derivedMaxCount == other.derivedMaxCount
           && maxPartitions == other.maxPartitions;
  }
};

struct Filters;

// The scene is organized with two separate accessors on the geometry data:
// - "views" are read-only and used at runtime. They may point to memory mapped files.
// - "storage" is read-write and used during processing time.
//   For larger scenes storage is typically discarded.

class Scene
{
public:
  //////////////////////////////////////////////////////////////////////////

  enum Result
  {
    SCENE_RESULT_SUCCESS,
    SCENE_RESULT_CACHE_INVALID,
    SCENE_RESULT_NEEDS_PREPROCESS,
    SCENE_RESULT_PREPROCESS_COMPLETED,
    SCENE_RESULT_ERROR,
  };

  Result init(const std::filesystem::path& filePath,
              const SceneConfig&           config,
              const SceneLoaderConfig&     loaderConfig,
              const std::string&           cacheSuffix,
              bool                         skipCache);
  bool   saveCache() const;
  void   deinit();

  void updateSceneGrid(const SceneGridConfig& gridConfig);

  // Assigns the instances to spatial partitions, see ScenePartitionConfig::Layout.
  // Result is cached, repeated calls with the same config and scene state do nothing.
  void updatePartitions(const ScenePartitionConfig& partitionConfig);
  // both fill `m_instancePartitions` and return the number of partitions used
  uint32_t buildPartitionsGrid(const ScenePartitionConfig& partitionConfig);
  uint32_t buildPartitionsBvh(const ScenePartitionConfig& partitionConfig);
  uint32_t buildPartitionsQuadtree(const ScenePartitionConfig& partitionConfig);
  uint32_t buildPartitionsCells(const ScenePartitionConfig& partitionConfig);
  uint32_t buildPartitionsNodeGroups();
  // cells per axis of the derived grid before its refinement, the quadtree starts from them
  uint32_t getPartitionGridBaseCells(const ScenePartitionConfig& partitionConfig) const;
  // xz position of the instance center in cells of a `perAxis` grid over the scene bounds
  glm::vec2 getPartitionGridPos(const shaderio::BBox& instanceBbox, uint32_t perAxis) const;
  // how many partitions the budget and the device limit allow
  uint32_t getPartitionBudget(const ScenePartitionConfig& partitionConfig) const;

  bool isMemoryMappedCache() const { return m_loadedFromCache && m_cacheFileMapping.valid(); }

  const std::filesystem::path& getFilePath() const { return m_filePath; }
  const std::filesystem::path& getCacheFilePath() const { return m_cacheFilePath; }


  //////////////////////////////////////////////////////////////////////////

  // Cluster Group

  struct Range
  {
    uint32_t offset;
    uint32_t count;
  };

  // To optimize streaming all cluster groups are stored in a contiguous blob of memory.
  struct GroupInfo
  {
    static constexpr size_t MAX_VERTEX_DATA_COUNT =
        // pos + nrm/tan + 2 * tex
        SHADERIO_MAX_GROUP_CLUSTERS * SHADERIO_MAX_CLUSTER_VERTICES * (3 + 1 + 2 * 2) +
        // + clusters for tex alignment
        SHADERIO_MAX_GROUP_CLUSTERS;
    static constexpr size_t MAX_TRIANGLE_DATA_COUNT = SHADERIO_MAX_GROUP_CLUSTERS * SHADERIO_MAX_CLUSTER_TRIANGLES * 4;

    static constexpr size_t MAX_SIZE =
        nvutils::align_up(sizeof(shaderio::Group) +
                              // clusters
                              (sizeof(shaderio::Cluster) + sizeof(shaderio::BBox) + sizeof(uint32_t)) * SHADERIO_MAX_GROUP_CLUSTERS +
                              // vertex
                              sizeof(float) * MAX_VERTEX_DATA_COUNT +
                              // triangle
                              sizeof(uint8_t) * MAX_TRIANGLE_DATA_COUNT,
                          16);

    uint64_t offsetBytes : 42;

    // MAX_SIZE
    uint64_t sizeBytes : 22;

    // SHADERIO_MAX_GROUP_CLUSTERS * SHADERIO_MAX_CLUSTER_VERTICES
    uint16_t vertexCount;
    // SHADERIO_MAX_GROUP_CLUSTERS * SHADERIO_MAX_CLUSTER_TRIANGLES
    uint16_t triangleCount;
    // SHADERIO_MAX_LOD_LEVELS
    uint32_t lodLevel : 6;
    // MAX_TRIANGLE_DATA_COUNT
    uint32_t triangleDataCount : 18;
    // SHADERIO_MAX_GROUP_CLUSTERS
    uint32_t clusterCount : 8;

    // MAX_VERTEX_DATA_COUNT
    uint64_t vertexDataCount : 21;
    // these must be 0 if group is stored 'uncompressed'
    // otherwise they provide the size information of the uncompressed state.
    uint64_t uncompressedVertexDataCount : 21;
    uint64_t uncompressedSizeBytes : 22;
    // MAX_TRIANGLE_DATA_COUNT, 0 if group is stored 'uncompressed'
    uint32_t uncompressedTriangleDataCount : 18;

    // compression may impact the size on device
    uint32_t getDeviceSize() const { return uint32_t(uncompressedSizeBytes ? uncompressedSizeBytes : sizeBytes); }

    // safe upper bound
    uint32_t estimateVertexDataCount(uint32_t attributeBits) const;
    uint32_t estimateTriangleDataCount(bool hasTriangleMaterials) const;

    // compute size based on relevant properties
    size_t computeSize() const;

    // compute size of the uncompressed section in a compressed group
    // based on relevant properties.
    // It is the leading part that a compressed group stores verbatim, everything
    // from the triangle region on is compressed.
    size_t computeUncompressedSectionSize() const;

    // byte offset (within the runtime group blob) where the vertex region begins,
    // that is behind the uncompressed section and the runtime triangle region
    size_t computeRuntimeVerticesOffset() const;

    // At runtime the trailing vertex region is laid out as [attributes][positions]:
    //   attributes = per-cluster normals/texcoords (attributesFloatCount() floats)
    //   positions  = per-cluster vec3 positions     (positionsFloatCount() floats, group tail)
    // number of float entries in the runtime (uncompressed) vertex region
    uint32_t getRuntimeVertexDataCount() const
    {
      return uint32_t(uncompressedVertexDataCount ? uncompressedVertexDataCount : vertexDataCount);
    }
    // number of bytes in the runtime (uncompressed) triangle region
    uint32_t getRuntimeTriangleDataCount() const
    {
      return uncompressedTriangleDataCount ? uncompressedTriangleDataCount : uint32_t(triangleDataCount);
    }
    // vec3 positions, one per vertex
    uint32_t positionsFloatCount() const { return 3u * uint32_t(vertexCount); }
    // normals/texcoords float count = everything in the vertex region except the trailing positions
    uint32_t attributesFloatCount() const { return getRuntimeVertexDataCount() - positionsFloatCount(); }
    // byte offset (within the runtime group blob) where the trailing positions region begins
    size_t positionsByteOffset() const
    {
      return computeRuntimeVerticesOffset() + size_t(attributesFloatCount()) * sizeof(float);
    }
    // byte size of the trailing positions region
    size_t positionsByteSize() const { return size_t(positionsFloatCount()) * sizeof(float); }
  };

  // read-only accessor of cluster groups used at runtime
  struct GroupView
  {
    const uint8_t*                     raw     = nullptr;
    const size_t                       rawSize = 0;
    const shaderio::Group*             group   = nullptr;
    std::span<const shaderio::Cluster> clusters;
    std::span<const uint32_t>          clusterGeneratingGroups;
    std::span<const shaderio::BBox>    clusterBboxes;
    std::span<const uint8_t>           triangles;
    std::span<const float>             vertices;

    GroupView() {};

    // input is array over all groupDatas
    GroupView(std::span<const uint8_t> groupDatas, const GroupInfo& info)
        : rawSize(info.sizeBytes)
    {
      assert(info.offsetBytes + info.sizeBytes <= groupDatas.size());
      raw = &groupDatas[info.offsetBytes];

      size_t startAddress = size_t(raw);

      group = (const shaderio::Group*)raw;
      clusters = std::span((const shaderio::Cluster*)nvutils::align_up(startAddress + sizeof(shaderio::Group), 16), info.clusterCount);
      clusterGeneratingGroups =
          std::span((const uint32_t*)nvutils::align_up(size_t(clusters.data() + info.clusterCount), 4), info.clusterCount);
      clusterBboxes =
          std::span((const shaderio::BBox*)nvutils::align_up(size_t(clusterGeneratingGroups.data() + info.clusterCount), 16),
                    info.clusterCount);

      triangles = std::span((const uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleDataCount);

      vertices = std::span((const float*)nvutils::align_up(size_t(triangles.data() + info.triangleDataCount), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }

    const uint8_t* getClusterIndices(size_t clusterIndex) const
    {
      // offsets relative to cluster header
      return (const uint8_t*)(size_t(&clusters[clusterIndex]) + shaderio::Cluster_getTrianglesOffset(clusters[clusterIndex]));
    }
    const glm::vec3* getClusterVertices(size_t clusterIndex) const
    {
      // offsets relative to cluster header
      return (const glm::vec3*)(size_t(&clusters[clusterIndex]) + shaderio::Cluster_getPositionsOffset(clusters[clusterIndex]));
    }
  };

  // read-write accessor used for processing a cluster group.
  // also used during compression.
  // same structure as above
  struct GroupStorage
  {
    uint8_t*                     raw;
    const size_t                 rawSize = 0;
    shaderio::Group*             group;
    std::span<shaderio::Cluster> clusters;
    std::span<uint32_t>          clusterGeneratingGroups;
    std::span<shaderio::BBox>    clusterBboxes;
    std::span<uint8_t>           triangles;
    std::span<float>             vertices;

    GroupStorage() {};

    // input is pointer to local groupData, does not apply info.offsetBytes!
    GroupStorage(void* groupData, const GroupInfo& info)
        : rawSize(info.sizeBytes)
    {
      size_t startAddress = (size_t)groupData;

      raw   = (uint8_t*)groupData;
      group = (shaderio::Group*)startAddress;
      clusters = std::span((shaderio::Cluster*)nvutils::align_up(startAddress + sizeof(shaderio::Group), 16), info.clusterCount);
      clusterGeneratingGroups =
          std::span((uint32_t*)nvutils::align_up(size_t(clusters.data() + info.clusterCount), 4), info.clusterCount);
      clusterBboxes =
          std::span((shaderio::BBox*)nvutils::align_up(size_t(clusterGeneratingGroups.data() + info.clusterCount), 16),
                    info.clusterCount);
      triangles = std::span((uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleDataCount);
      vertices = std::span((float*)nvutils::align_up(size_t(triangles.data() + info.triangleDataCount), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }

    // cluster data pointers are stored as offsets relative to the Cluster's header.
    uint32_t getClusterLocalOffset(uint32_t clusterIndex, const void* input, size_t overrideSize = 0) const
    {
      assert(size_t(input) >= size_t(&clusters[clusterIndex]));
      assert(size_t(input) < size_t(raw + (overrideSize ? overrideSize : rawSize)));

      return uint32_t(size_t(input) - size_t(&clusters[clusterIndex]));
    }

    // get pointer relative to cluster header
    uint32_t* getClusterLocalData(uint32_t clusterIndex, uint32_t localOffset)
    {
      return (uint32_t*)(size_t(&clusters[clusterIndex]) + localOffset);
    }
  };


  // used for preloaded groups, streamed in groups are patched in shaders.
  // scratch decode space (see decompressGroup)
  static void fillGroupRuntimeData(const GroupInfo&       srcGroupInfo,
                                   const GroupView&       srcGroupView,
                                   uint32_t               groupID,
                                   uint32_t               groupResidentID,
                                   uint32_t               clusterResidentID,
                                   void*                  dst,
                                   size_t                 dstSize,
                                   std::vector<uint32_t>& scratch);

  // used to decompress group on CPU.
  // typically write-combined memory destination.
  // scratch decode space, reused across calls (decompressGroup grows it as needed)
  static void decompressGroup(const GroupInfo& info, const GroupView& groupView, void* dstWriteOnly, size_t dstSize, std::vector<uint32_t>& scratch);

  // decodes the per-triangle material bytes of a group. `triangleMaterials` must hold
  // `info.triangleCount` bytes, `perCluster` `info.clusterCount` pointers into it, which are
  // null for clusters without such materials.
  static void getGroupTriangleMaterials(const GroupInfo& info, const GroupView& groupView, uint8_t* triangleMaterials, const uint8_t** perCluster);


  //////////////////////////////////////////////////////////////////////////

  // Geometry

  struct GeometryLodInput
  {
    uint64_t inputTriangleCount       = 0;
    uint64_t inputVertexCount         = 0;
    uint64_t inputTriangleIndicesHash = 0;
    uint64_t inputVerticesHash        = 0;
    // local material slot partition and the material properties baked into the
    // cluster/triangle state bits. Invalidates the cache when materials change.
    uint64_t inputMaterialSetHash = 0;
    // the `SimplifyOverrides` entries that applied to this geometry, 0 when none did.
    // Invalidates the cache when the override file changes.
    uint64_t inputSimplifyOverrideHash = 0;

    // this struct is compared as raw bytes against the cache file, so new inputs must come
    // out of this section and default to zero. That keeps caches of scenes that don't use
    // the new input valid, without another `geoVersion` bump.
    uint64_t reservedData[6] = {};
  };

  struct GeometryBase
  {
    uint32_t attributeBits = 0;

    uint32_t clusterMaxVerticesCount{};
    uint32_t clusterMaxTrianglesCount{};

    uint32_t lodLevelsCount{};

    // based on highest detail lod
    uint32_t hiTriangleCount{};
    uint32_t hiVerticesCount{};
    uint32_t hiClustersCount{};

    // total sum
    uint32_t totalTriangleCount{};
    uint32_t totalVerticesCount{};
    uint32_t totalClustersCount{};

    shaderio::BBox bbox{};

    // oriented 26-DOP over the same positions as `bbox`, host only.
    // source for the per-instance AABBs, see `Scene::m_geometryHulls`
    KDop kdop{};

    GeometryLodInput lodInfo;

    uint32_t instanceReferenceCount{};

    // Lowest-detail cluster `shaderio::Cluster::stateBits` after LOD build (single last-LOD cluster).
    uint8_t lowDetailClusterStateBits{};
  };

  // read-only accessor for the geometry data.
  // used at runtime.
  struct GeometryView : GeometryBase
  {
    // may contain compressed or uncompressed data
    std::span<const uint8_t> groupData;

    // info about state of a group
    std::span<const GroupInfo> groupInfos;
    // packed overlap table: per-group header + flat coarse-then-fine entries
    std::span<const shaderio::GroupOverlapInfo> groupOverlapInfos;
    std::span<const uint32_t>                   groupOverlapEntries;

    std::span<const shaderio::LodLevel> lodLevels;
    std::span<const shaderio::Node>     lodNodes;
    std::span<const shaderio::BBox>     lodNodeBboxes;

    // if we have multiple material IDs
    std::span<const uint32_t> localMaterialIDs;

    inline uint64_t getCachedSize() const
    {
      uint64_t cachedSize = 0;

      cachedSize += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;
      cachedSize += serialization::getCachedSize(groupData);
      cachedSize += serialization::getCachedSize(groupInfos);
      cachedSize += serialization::getCachedSize(groupOverlapInfos);
      cachedSize += serialization::getCachedSize(groupOverlapEntries);
      cachedSize += serialization::getCachedSize(lodLevels);
      cachedSize += serialization::getCachedSize(lodNodes);
      cachedSize += serialization::getCachedSize(lodNodeBboxes);
      cachedSize += serialization::getCachedSize(localMaterialIDs);

      return cachedSize;
    }
  };

  // we virtually instance geometries to avoid higher cpu memory consumption
  // happens when the grid config is larger
  const GeometryView& getActiveGeometry(size_t idx) const { return m_geometryViews[m_activeToOriginal[idx]]; }
  size_t              getActiveGeometryCount() const { return m_activeToOriginal.size(); }
  size_t              getOriginalGeometryIndex(size_t idx) const { return m_activeToOriginal[idx]; }

  // transform these by an instance matrix and take the min/max for a tight world AABB,
  // see `kdopBuildHull` for why it has to be these points
  const KDopHull& getActiveGeometryHull(size_t idx) const { return m_geometryHulls[m_activeToOriginal[idx]]; }

  // how many grid copies reference this active geometry
  uint32_t getActiveGeometryInstanceFactor(size_t idx) const { return m_activeInstanceFactor[idx]; }


  //////////////////////////////////////////////////////////////////////////

  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
    // slot 0 of the material set, kept for the single-material fast path
    uint32_t materialID = ~0U;
    // index into m_instanceMaterialSets, provides the materials for all local slots
    uint32_t  materialSetID = ~0U;
    glm::vec4 color{0.8, 0.8, 0.8, 1.0f};
  };

  // world-space AABB of an instance, from its geometry's k-DOP hull.
  // tighter than transforming `GeometryBase::bbox` and never looser.
  shaderio::BBox getInstanceWorldBBox(const Instance& instance) const;

  // geometries are deduplicated across glTF meshes that only differ in materials,
  // so the actual material per local slot comes from the instance, not the geometry.
  struct MaterialSetRange
  {
    uint32_t offset = 0;
    uint32_t count  = 0;
  };

  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };

  enum ImageDefaultType
  {
    IMAGE_DEFAULT_WHITE,
    IMAGE_DEFAULT_BLACK,
    IMAGE_DEFAULT_NORMAL,
    NUM_IMAGE_DEFAULTS,
  };

  // what the shader reads the image as; formats missing those channels get an image view swizzle
  enum ImageChannelLayout
  {
    IMAGE_CHANNELS_DEFAULT,             // sampled as stored
    IMAGE_CHANNELS_METALLIC_ROUGHNESS,  // glTF: roughness in G, metalness in B
    IMAGE_CHANNELS_SPECULAR,            // KHR_materials_specular: strength in A
  };

  struct Image
  {
    std::string        filename;
    bool               sRGB          = false;
    ImageDefaultType   defaultType   = IMAGE_DEFAULT_NORMAL;
    ImageChannelLayout channelLayout = IMAGE_CHANNELS_DEFAULT;
  };

  struct Material
  {
    bool  twoSided    = false;
    bool  alphaMasked = false;
    float alphaCutOff = 0.5f;

    glm::vec4 color{0, 0, 0, 1};
    glm::vec4 emissive{0, 0, 0, 1};
    float     metallicFactor           = 1.0;
    float     roughnessFactor          = 1.0;
    uint32_t  normalImageID            = ~0u;
    uint32_t  baseImageID              = ~0u;
    uint32_t  occlusionImageID         = ~0u;
    uint32_t  metallicRoughnessImageID = ~0u;
    uint32_t  emissiveImageID          = ~0u;
    uint32_t  alphaMaskImageID         = ~0u;
    float     specularFactor           = 1.0f;
    glm::vec3 specularColorFactor{1, 1, 1};
    uint32_t  specularImageID      = ~0u;
    uint32_t  specularColorImageID = ~0u;
  };

  //////////////////////////////////////////////////////////////////////////

  // statistics

  struct Histograms
  {
    static const uint32_t version = 1;

    std::array<uint32_t, SHADERIO_MAX_CLUSTER_TRIANGLES + 1> clusterTriangles = {};
    std::array<uint32_t, SHADERIO_MAX_CLUSTER_VERTICES + 1>  clusterVertices  = {};
    std::array<uint32_t, SHADERIO_MAX_GROUP_CLUSTERS + 1>    groupClusters    = {};
    std::array<uint32_t, SHADERIO_MAX_NODE_CHILDREN + 1>     nodeChildren     = {};
    std::array<uint32_t, SHADERIO_MAX_LOD_LEVELS + 1>        lodLevels        = {};

    uint32_t clusterTrianglesMax = {};
    uint32_t clusterVerticesMax  = {};
    uint32_t groupClustersMax    = {};
    uint32_t nodeChildrenMax     = {};
    uint32_t lodLevelsMax        = {};
  };

  //////////////////////////////////////////////////////////////////////////

  SceneConfig       m_config;
  SceneLoaderConfig m_loaderConfig;
  SceneGridConfig   m_gridConfig;
  std::string       m_cacheSuffix;

  shaderio::BBox m_bbox;
  shaderio::BBox m_gridBbox;

  std::vector<Instance> m_instances;

  // spatial partitions, see `updatePartitions`
  struct Partition
  {
    shaderio::BBox bbox;
    uint32_t       instanceCount;
    // first entry in `m_partitionInstanceOrder`, membership is fixed so this is built once
    uint32_t instanceOffset;
  };
  ScenePartitionConfig  m_partitionConfig;
  shaderio::BBox        m_partitionSceneBbox{};
  std::vector<uint32_t> m_instancePartitions;
  // instance ids grouped by partition, so a partition owns a contiguous range
  std::vector<uint32_t>  m_partitionInstanceOrder;
  std::vector<Partition> m_partitions;
  // number of partitions with n instances, the last bin also counts larger ones
  static constexpr uint32_t PARTITION_HISTOGRAM_MAX = 4096;
  std::vector<uint32_t>     m_partitionHistogram;
  uint32_t                  m_partitionHistogramMax = 0;
  uint32_t                  m_partitionsPerAxis     = 0;
  // per original instance, the node group from `SceneLoaderConfig::partitionNodeNames`,
  // ~0 when no matching node is above it. Empty when the option is not used.
  std::vector<uint32_t> m_instanceNodeGroups;
  uint32_t              m_nodeGroupCount = 0;
  // original instances below no matching node, they share one partition
  size_t m_nodeGroupUngroupedCount = 0;

  // deduplicated material sets, one entry per distinct set (bounded by mesh count)
  std::vector<MaterialSetRange> m_instanceMaterialSets;
  // flat pool of scene material IDs the ranges above point into
  std::vector<uint32_t>    m_instanceMaterialSetData;
  std::vector<Camera>      m_cameras;
  std::vector<Material>    m_materials;
  std::vector<std::string> m_geometryNames;

  // parsed from `SceneLoaderConfig::simplifyOverridesFile`, empty when none was given
  SimplifyOverrides        m_simplifyOverrides;
  std::vector<std::string> m_materialNames;
  std::vector<Image>       m_images;

  bool m_isBig                = false;
  bool m_hasTwoSided          = false;
  bool m_hasAlphaMask         = false;
  bool m_hasTexturedMaterials = false;

  // maxima across lod levels
  uint32_t m_maxPerGeometryClusters  = 0;
  uint32_t m_maxPerGeometryTriangles = 0;
  uint32_t m_maxPerGeometryVertices  = 0;
  uint32_t m_maxPerGeometryGroups    = 0;

  uint32_t m_maxClusterTriangles = 0;
  uint32_t m_maxClusterVertices  = 0;
  uint32_t m_maxLodLevelsCount   = 0;
  uint32_t m_maxNodeTreeDepth    = 0;

  // maxima in lod 0
  uint32_t m_hiPerGeometryClusters  = 0;
  uint32_t m_hiPerGeometryTriangles = 0;
  uint32_t m_hiPerGeometryVertices  = 0;
  uint32_t m_hiPerGeometryGroups    = 0;

  // sum in lod 0
  uint64_t m_hiClustersCount           = 0;
  uint64_t m_hiTrianglesCount          = 0;
  uint64_t m_hiClustersCountInstanced  = 0;
  uint64_t m_hiTrianglesCountInstanced = 0;

  // sum across all lod levels
  uint64_t m_totalClustersCount  = 0;
  uint64_t m_totalTrianglesCount = 0;
  uint64_t m_totalVerticesCount  = 0;

  Histograms m_histograms;

  bool m_loadedFromCache    = false;
  bool m_hasVertexNormals   = false;
  bool m_hasVertexTexCoord0 = false;
  bool m_hasVertexTexCoord1 = false;
  bool m_hasVertexTangents  = false;

  size_t m_originalInstanceCount = 0;
  size_t m_originalGeometryCount = 0;

  size_t m_cacheFileSize = 0;

private:
  //////////////////////////////////////////////////////////////////////////

  // Geometry

  // read-write accessor to Geometry. Allows building and modifying the data in system RAM
  struct GeometryStorage : GeometryBase
  {
    // temporary, removed after processing
    std::vector<glm::vec3>  vertexPositions;
    std::vector<float>      vertexAttributes;
    std::vector<glm::uvec3> triangles;

    // `SceneConfig` with this mesh's `SimplifyOverrides` already applied, used by the
    // lod build. Equal to `Scene::m_config` when no override matched.
    SceneConfig lodConfig;

    uint32_t attributesWithWeights   = 0u;
    uint32_t attributeNormalOffset   = ~0u;
    uint32_t attributeTex0offset     = ~0u;
    uint32_t attributeTex1offset     = ~0u;
    uint32_t attributeTangentOffset  = ~0u;
    uint32_t attributeMaterialOffset = ~0u;

    // persistent used in view
    std::vector<uint8_t>                    groupData;
    std::vector<GroupInfo>                  groupInfos;
    std::vector<shaderio::GroupOverlapInfo> groupOverlapInfos;
    std::vector<uint32_t>                   groupOverlapEntries;

    std::vector<shaderio::LodLevel> lodLevels;
    std::vector<shaderio::BBox>     lodNodeBboxes;
    std::vector<shaderio::Node>     lodNodes;

    std::vector<uint32_t> localMaterialIDs;
  };

  // active geometry id -> original geometry, grid copies with unique geometries append new ids
  std::vector<uint32_t> m_activeToOriginal;
  std::vector<uint32_t> m_activeInstanceFactor;

  std::vector<GeometryStorage> m_geometryStorages;
  std::vector<GeometryView>    m_geometryViews;

  // derived from the geometries' `kdop` at load, never cached
  std::vector<KDopHull> m_geometryHulls;

  //////////////////////////////////////////////////////////////////////////

  // Cache File

  static bool     loadCached(GeometryView& view, uint64_t dataSize, const void* data);
  static bool     storeCached(const GeometryView& view, uint64_t dataSize, void* data);
  static uint64_t storeCached(const GeometryView& view, FILE* outFile);

  void openCache();
  void closeCache();

  bool checkCache(const GeometryLodInput& info, size_t geometryIndex);
  void loadCachedGeometry(GeometryStorage& geometry, size_t geometryIndex);

  class CacheFileHeader
  {
  public:
    CacheFileHeader()
    {
      memset(this, 0, sizeof(CacheFileHeader));
      header = {};
      config = {};
    }

    bool isValid() const
    {
      Header reference = {};

      return header.magic == reference.magic && header.geoVersion == reference.geoVersion
             && header.geoStructSize == reference.geoStructSize && header.configStructSize == reference.configStructSize
             && header.alignment == reference.alignment;
    }

  private:
    struct Header
    {
      uint64_t magic               = 0x006f65676e73766eULL;  // nvsngeo
      uint32_t geoVersion          = 16;
      uint32_t geoStructSize       = uint32_t(sizeof(GeometryView));
      uint32_t configVersion       = SceneConfig::version;
      uint32_t configStructSize    = uint32_t(sizeof(SceneConfig));
      uint32_t histogramsVersion   = Histograms::version;
      uint32_t histogramStructSize = uint32_t(sizeof(Histograms));
      uint64_t alignment           = serialization::ALIGNMENT;

      // geoVersion history:
      // 1 initial
      // 2 bugfix wrong storage of `lodInfo`
      // 3 octant vertices
      // 4 table is 2 x 64-bit per geometry (offset + size) to allow out of order storage
      // 5
      // 6 reduced shaderio::Group/Cluster structs using relative offsets
      // 7 compression
      // 8 triangle data
      // 9 GeometryBase.lowDetailClusterStateBits
      // 10 cluster positions split into a trailing group region ([attributes][positions]),
      //    shaderio::Cluster offsets packed into 3x24-bit fields
      // 11 GeometryLodInput.inputMaterialSetHash, geometry dedup keyed by material partition
      // 12 bugfix texcoord compressor header size (was hardcoded 32*3, now 32*DIM)
      // 13 GeometryLodInput.inputSimplifyOverrideHash, plus reserved room so further
      //    lod inputs no longer need a version bump.
      //    GroupInfo.uncompressedTriangleDataCount, compressed groups encode their
      //    triangle indices with meshoptimizer's meshlet codec
      // 14 compressed groups palette encode their per-triangle material bytes
      // 15 GeometryBase.kdop, oriented 26-DOP per geometry
      // 16 group overlap tables for streaming (GroupOverlapInfo + flat entries, both directions)
    };

    Header header;

  public:
    SceneConfig config;
    Histograms  histograms;
    uint32_t    pad[7];
  };

  static_assert(sizeof(CacheFileHeader) % serialization::ALIGNMENT == 0, "CacheFileHeader size unaligned");

  class CacheFileView
  {
    // Optionally if you want to have a simple cache file for this
    // data, we provide a canonical layout, and this simple class
    // to open it.
    //
    // The cache data must be stored in three sections:
    //
#if 0
    struct CacheFile
    {
      // first: library version specific header
      CacheHeader header;
      // second: for each geometry serialized data of the `LodGeometryView`
      uint8_t geometryViewData[];
      // third: offset table
      // offsets where each `LodGeometry` data is stored + size
      // ordered with ascending offsets
      // `geometryDataSize = geometryOffsets[geometryIndex * 2 + 1];`
      uint64_t geometryOffsets[geometryCount * 2];
      uint64_t geometryCount;
    };
#endif

  public:
    bool isValid() const { return m_dataSize != 0; }

    bool init(uint64_t dataSize, const void* data);

    void deinit() { *(this) = {}; }

    uint64_t getGeometryCount() const { return m_geometryCount; }

    void getSceneConfig(SceneConfig& settings) const;
    void getHistograms(Histograms& histograms) const;

    bool getGeometryView(GeometryView& view, uint64_t geometryIndex) const;

  private:
    template <class T>
    const T* getPointer(uint64_t offset, uint64_t count = 1) const
    {
      assert(offset + sizeof(T) * count <= m_dataSize);
      return reinterpret_cast<const T*>(m_dataBytes + offset);
    }

    uint64_t       m_dataSize      = 0;
    uint64_t       m_tableStart    = 0;
    const uint8_t* m_dataBytes     = nullptr;
    uint64_t       m_geometryCount = 0;
  };

  struct CachePartialEntry
  {
    uint64_t geometryIndex = 0;
    uint64_t offset        = 0;
    uint64_t dataSize      = 0;
  };

  std::filesystem::path m_filePath;
  std::filesystem::path m_cacheFilePath;
  std::filesystem::path m_cachePartialFilePath;

  // When loading a scene from a cache file, we can actually
  // directly load all data from the memory mapped file, rather than
  // copying it into system memory.
  // This view and mapping are kept alive after init when
  // `SceneConfig::memoryMappedCache` is true, otherwise they are closed
  // within `Scene::init`.

  nvutils::FileReadMapping m_cacheFileMapping;
  CacheFileView            m_cacheFileView;

  //////////////////////////////////////////////////////////////////////////

  // Processing

  // only used in `processingOnly` mode
  FILE*                 m_processingOnlyFile        = nullptr;
  FILE*                 m_processingOnlyPartialFile = nullptr;
  uint64_t              m_processingOnlyFileOffset  = 0;
  std::vector<uint64_t> m_processingOnlyGeometryOffsets;

  struct ProcessingInfo
  {
    // multi-threading is done over geometries, a single geometry is processed serially

    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;

    // if triangleCount is not 0, then we will track progress
    // based on completed triangles, otherwise based on
    // completed geometries
    size_t   geometryCount = 0;
    uint64_t triangleCount = 0;

    std::mutex processOnlySaveMutex;

    // Admission control over the outer threads. Geometry sizes are heavily skewed and
    // the large-first ordering puts the biggest ones in-flight together, which is what
    // sets the peak. This throttles on estimated memory rather than thread count, so
    // small geometries still saturate the threads.
    struct MemoryBudget
    {
      std::mutex              mutex;
      std::condition_variable condition;
      // 0 disables throttling
      uint64_t budgetBytes   = 0;
      uint64_t inFlightBytes = 0;
      uint32_t inFlightCount = 0;
      // stats
      uint64_t peakInFlightBytes = 0;
      uint32_t peakInFlightCount = 0;
      uint32_t waitCount         = 0;

      void acquire(uint64_t bytes);
      void release(uint64_t bytes);
    } memoryBudget;

    // estimated peak memory for processing a geometry of this size
    static uint64_t estimateGeometryProcessingBytes(uint64_t triangleCount);
    // resolves `SceneLoaderConfig::processingMemoryGiB` against system memory
    void setupMemoryBudget(int budgetGiB);

    // bufferview compression

    std::vector<uint32_t> bufferViewUsers;
    std::vector<uint32_t> bufferViewLocks;

    // stats

    struct Stats
    {
      std::atomic_uint64_t groups                  = 0;
      std::atomic_uint64_t clusters                = 0;
      std::atomic_uint64_t multiMaterialClusters   = 0;
      std::atomic_uint64_t vertices                = 0;
      std::atomic_uint64_t groupUniqueVertices     = 0;
      std::atomic_uint64_t groupHeaderBytes        = 0;
      std::atomic_uint64_t triangleIndexBytes      = 0;
      std::atomic_uint64_t triangleDataBytes       = 0;
      std::atomic_uint64_t vertexPosBytes          = 0;
      std::atomic_uint64_t vertexTexCoordBytes     = 0;
      std::atomic_uint64_t vertexNrmBytes          = 0;
      std::atomic_uint64_t vertexCompressedBytes   = 0;
      std::atomic_uint64_t triangleCompressedBytes = 0;
      std::atomic_uint64_t clusterBboxBytes        = 0;
      std::atomic_uint64_t clusterHeaderBytes      = 0;
      std::atomic_uint64_t clusterGenBytes         = 0;
    } stats;


    // logging progress

    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    uint64_t   progressTrianglesCompleted  = 0;
    std::mutex progressMutex;

    // geometries whose data breaks an invariant the shaders rely on, fails the load
    std::atomic_uint32_t invalidGeometries = 0;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;

    void init(float pct);
    void setupParallelism(size_t geometryCount_);
    void setupCompressedGltf(size_t bufferViewCount);
    void deinit();

    void     logBegin(uint64_t totalTriangleCount);
    uint64_t logCompletedGeometry(uint64_t triangleCount = 0);
    void     logEnd();
  };

  Result loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath);

private:
  void loadGeometryGLTF(ProcessingInfo&          processingInfo,
                        uint64_t                 geometryIndex,
                        size_t                   meshIndex,
                        uint64_t                 materialSetHash,
                        const struct cgltf_data* gltf);
  void addInstancesFromNodeGLTF(const std::vector<size_t>&   meshToGeometry,
                                const std::vector<uint32_t>& meshToMaterialSet,
                                const struct cgltf_data*     data,
                                const struct cgltf_node*     node,
                                const glm::mat4              parentObjToWorldTransform,
                                struct Filters*              filters   = nullptr,
                                uint32_t                     nodeGroup = ~0u);

  // to handle glTF EXT_meshopt_compression
  bool loadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                               std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                               const struct cgltf_data*                       gltf);
  void unloadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                 std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                                 const struct cgltf_data*                       gltf);

  void processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached);

  void buildGeometryLod(ProcessingInfo& processingInfo, GeometryStorage& geometry);
  void buildGeometryLodHierarchy(GeometryStorage& geometry);
  void buildGeometryLodOverlap(ProcessingInfo& processingInfo, GeometryStorage& geometry);

  void computeLodBboxes_recursive(GeometryStorage& geometry, size_t nodeIdx);
  void buildGeometryDedupVertices(ProcessingInfo& processingInfo, GeometryStorage& geometry);

  void computeHistogramMaxs();
  void computeInstanceBBoxes();

  // these modes always output to the cache directly
  bool beginProcessingOnly(size_t geometryCount);
  void saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex);
  bool endProcessingOnly(bool hadError);


  //////////////////////////////////////////////////////////////////////////

  // Cluster Group Building

  struct TempContext
  {
    ProcessingInfo&  processingInfo;
    GeometryStorage& geometry;
    Scene&           scene;

    GroupInfo tempGroupInfo        = {};
    uint32_t  tempGroupSize        = 0;
    uint32_t  tempGroupStorageSize = 0;
    uint32_t  lodLevel             = ~0u;

    // scratch for the group currently being assembled
    std::vector<uint8_t> tempGroupData;
    // scratch for the triangle index compression, see `compressGroupTriangles`
    std::vector<uint8_t> tempTriangleData;
  };

  struct TempGroup
  {
    uint32_t                  lodLevel;
    uint32_t                  clusterCount;
    shaderio::TraversalMetric traversalMetric;
  };

  struct TempCluster
  {
    const uint32_t* indices         = nullptr;
    uint32_t        indexCount      = 0;
    uint32_t        generatingGroup = 0;
  };

  void applyMaterialStateBits(uint32_t& stateBits, const GeometryStorage& geometry, uint32_t localMaterialID, bool isFirst);
  void applyMaterialStateBits(uint32_t& stateBits, uint32_t clusterBits, bool isFirst);
  void applyMaterialTriangleBits(uint8_t& triangleBits, const GeometryStorage& geometry, uint32_t localMaterialID);

  static uint8_t getMaterialLocalIndex(const GeometryStorage& geometry, uint32_t index, uint32_t attributeStride);

  uint32_t storeGroup(TempContext* context, const clodGroup& group, uint32_t clusterCount, const clodCluster* clusters);

  void compressGroup(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo, uint32_t* vertexCacheLocal);
  void compressGroupTriangles(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo);

  static void decompressGroupTriangles(const GroupInfo& info, const GroupView& groupSrc, GroupStorage& groupDst);
  static int clodGroupMeshoptimizer(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count);
};

}  // namespace lodclusters
