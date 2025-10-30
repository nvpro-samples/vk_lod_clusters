/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <functional>

#include <glm/glm.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/alignment.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy_storage.hpp>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <nvclusterlod/nvclusterlod_cache.hpp>

#include "serialization.hpp"
#include "meshopt_clusterlod.h"
#include "../shaders/shaderio_scene.h"

namespace lodclusters {

struct SceneConfig
{
  static const uint32_t version = 1;

  // cluster and cluster group settings
  uint32_t clusterVertices    = 128;
  uint32_t clusterTriangles   = 128;
  uint32_t clusterGroupSize   = 32;
  uint32_t preferredNodeWidth = 8;

  // uses nv_cluster_lod_builder library,
  // will be deprecated in future.
  bool useNvLib = false;

  // default setting should prefer ray tracing
  bool meshoptPreferRayTracing = true;

  // not yet implemented, but add to make cache files binary compatible
  bool useCompressedData = false;

  uint32_t enabledAttributes = shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;

  // settings that affect clusterization
  float meshoptFillWeight  = 0.5f;  // if ray-tracing is preferred
  float meshoptSplitFactor = 2.0f;  // otherwise

  // at each lod step reduce cluster group triangles by this factor
  float lodLevelDecimationFactor = 0.5f;

  // lod error propagation for meshoptimizer's clusterlod
  // These control the error propagation across lod levels to
  // account for simplifying an already simplified mesh.
  // To get closer to nv_cluster_lod_builder:
  // previous 1.0f, additive 0.7f
  // but causes about 50% more triangles during traversal.
  float lodErrorMergePrevious = 1.5;
  float lodErrorMergeAdditive = 0.0f;

  // mesh simplification weights for attributes
  // zero to disable
  float simplifyNormalWeight      = 1.0f;
  float simplifyTangentWeight     = 0.01f;
  float simplifyTangentSignWeight = 0.5f;
  float simplifyUvWeight          = 0;

  //
  uint32_t reservedData[17] = {};
};

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
  // -1 inner, +1 outer, 0 auto
  int processingMode = 0;

  // save cache file after load automatically
  bool autoSaveCache = true;
  // try load from cache file if file was found
  bool autoLoadCache = true;
  // when loading from cache file, memory map it,
  // rather than loading it into system RAM.
  bool memoryMappedCache = false;

  // if a scenes geometry data exceeds this, then always do a separate preprocess pass
  // and use the cache file afterwards
  size_t forcePreprocessMiB = size_t(2) * 1024;

  // very crude metric to judge how "dense" a cluster is filled.
  // using the area of triangles divided by area of bbox sides
  bool computeClusterBBoxOccupancy = false;

  // optional thread-safe progress bar updates
  std::atomic_uint32_t* progressPct = nullptr;
};

struct SceneGridConfig
{
  // when set to true each new set of instance on the grid gets
  // its own unique set of geometries. This stresses the streaming system
  // and memory consumption a lot more.
  bool      uniqueGeometriesForCopies = false;
  uint32_t  numCopies                 = 1;
  uint32_t  gridBits                  = 13;
  glm::vec3 refShift                  = {1.0f, 1.0f, 1.0f};
  float     snapAngle                 = 0;
  float     minScale                  = 1.0f;
  float     maxScale                  = 1.0f;
};

class Scene
{
public:
  struct Range
  {
    uint32_t offset;
    uint32_t count;
  };

  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
    uint32_t       materialID = ~0U;
    glm::vec4      color{0.8, 0.8, 0.8, 1.0f};
  };

  struct GroupInfo
  {
    uint64_t offsetBytes : 42;
    uint64_t sizeBytes : 22;
    uint16_t vertexCount;
    uint16_t triangleCount;
    uint8_t  lodLevel;
    uint8_t  clusterCount;
    uint8_t  attributeBits;
    uint8_t  reserved1             = 0;
    uint16_t vertexDataCount       = 0;
    uint16_t reserved2             = 0;
    uint32_t uncompressedSizeBytes = 0;

    // safe upper bound
    uint32_t estimateVertexDataCount() const
    {
      uint32_t dataCount = vertexCount * 3;
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
      {
        dataCount += vertexCount * 1;
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_UV)
      {
        dataCount += vertexCount * 2;
        dataCount += clusterCount;  // potential padding
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
      {
        // 16 bit per entry
        dataCount += (vertexCount + 1) / 2;
        dataCount += clusterCount;  // potential padding
      }
      return dataCount;
    }

    size_t computeSize() const
    {
      size_t threadGroupSize = sizeof(shaderio::Group);
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::Cluster) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 4) + sizeof(uint32_t) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::BBox) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(float) * vertexDataCount;
      threadGroupSize        = threadGroupSize + sizeof(uint8_t) * triangleCount * 3;
      return nvutils::align_up(threadGroupSize, 16);
    }
  };

  struct GroupView
  {
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
      vertices = std::span((const float*)nvutils::align_up(size_t(clusterBboxes.data() + info.clusterCount), 16), info.vertexDataCount);
      indices = std::span((const uint8_t*)size_t(vertices.data() + info.vertexDataCount), info.triangleCount * 3);
      assert((size_t(indices.data() + indices.size()) - startAddress) <= size_t(info.sizeBytes));
    }

    const uint8_t*                     raw     = nullptr;
    const size_t                       rawSize = 0;
    const shaderio::Group*             group   = nullptr;
    std::span<const shaderio::Cluster> clusters;
    std::span<const uint32_t>          clusterGeneratingGroups;
    std::span<const shaderio::BBox>    clusterBboxes;
    std::span<const float>             vertices;
    std::span<const uint8_t>           indices;

    const uint8_t* getClusterIndices(size_t clusterIndex) const
    {
      // offsets relative to cluster header
      return (const uint8_t*)(size_t(&clusters[clusterIndex]) + clusters[clusterIndex].indices);
    }
    const glm::vec3* getClusterVertices(size_t clusterIndex) const
    {
      // offsets relative to cluster header
      return (const glm::vec3*)(size_t(&clusters[clusterIndex]) + clusters[clusterIndex].vertices);
    }
  };

  struct GroupStorage
  {
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
      vertices = std::span((float*)nvutils::align_up(size_t(clusterBboxes.data() + info.clusterCount), 16), info.vertexDataCount);
      indices = std::span((uint8_t*)size_t(vertices.data() + info.vertexDataCount), info.triangleCount * 3);
      assert((size_t(indices.data() + indices.size()) - startAddress) <= size_t(info.sizeBytes));
    }

    // cluster data pointers are stored as offsets relative to the Cluster's header.
    uint32_t getClusterLocalOffset(uint32_t clusterIndex, const void* input) const
    {
      assert(size_t(input) >= size_t(&clusters[clusterIndex]));
      assert(size_t(input) < size_t(raw + rawSize));

      return uint32_t(size_t(input) - size_t(&clusters[clusterIndex]));
    }

    uint8_t*                     raw;
    const size_t                 rawSize = 0;
    shaderio::Group*             group;
    std::span<shaderio::Cluster> clusters;
    std::span<uint32_t>          clusterGeneratingGroups;
    std::span<shaderio::BBox>    clusterBboxes;
    std::span<float>             vertices;
    std::span<uint8_t>           indices;
  };

  struct GeometryLodInput
  {
    uint64_t inputTriangleCount       = 0;
    uint64_t inputVertexCount         = 0;
    uint64_t inputTriangleIndicesHash = 0;
    uint64_t inputVerticesHash        = 0;
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

    GeometryLodInput lodInfo;

    uint32_t instanceReferenceCount{};
  };

  // not fully leveraged yet, but preparation to have a readonly view on
  // data accessed from persistently memory mapped files
  struct GeometryView : GeometryBase
  {
    // may contain compressed or uncompressed data
    std::span<const uint8_t> groupData;

    // info about state of a group
    std::span<const GroupInfo> groupInfos;

    std::span<const shaderio::LodLevel> lodLevels;
    std::span<const shaderio::Node>     lodNodes;
    std::span<const shaderio::BBox>     lodNodeBboxes;

    // if we have
    std::span<const uint32_t> localMaterialIDs;

    inline uint64_t getCachedSize() const
    {
      uint64_t cachedSize = 0;

      cachedSize += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;
      cachedSize += serialization::getCachedSize(groupData);
      cachedSize += serialization::getCachedSize(groupInfos);
      cachedSize += serialization::getCachedSize(lodLevels);
      cachedSize += serialization::getCachedSize(lodNodes);
      cachedSize += serialization::getCachedSize(lodNodeBboxes);
      cachedSize += serialization::getCachedSize(localMaterialIDs);

      return cachedSize;
    }
  };

  // used for preloaded groups, streamed in groups are patched in shaders.
  static void fillGroupRuntimeData(const GeometryView& sceneGeometry,
                                   uint32_t            groupID,
                                   uint32_t            groupResidentID,
                                   uint32_t            clusterResidentID,
                                   void*               dst,
                                   size_t              dstSize);

  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };

  enum Result
  {
    SCENE_RESULT_SUCCESS,
    SCENE_RESULT_CACHE_INVALID,
    SCENE_RESULT_NEEDS_PREPROCESS,
    SCENE_RESULT_PREPROCESS_COMPLETED,
    SCENE_RESULT_ERROR,
  };

  Result init(const std::filesystem::path& filePath, const SceneConfig& config, const SceneLoaderConfig& loaderConfig, bool skipCache);
  bool saveCache() const;
  void deinit();

  void updateSceneGrid(const SceneGridConfig& gridConfig);

  bool isMemoryMappedCache() const { return m_loadedFromCache && m_cacheFileMapping.valid(); }

  const std::filesystem::path& getFilePath() const { return m_filePath; }
  const std::filesystem::path& getCacheFilePath() const { return m_cacheFilePath; }

  SceneConfig       m_config;
  SceneLoaderConfig m_loaderConfig;
  SceneGridConfig   m_gridConfig;

  shaderio::BBox m_bbox;
  shaderio::BBox m_gridBbox;

  std::vector<Instance> m_instances;
  std::vector<Camera>   m_cameras;

  // we virtually instance geometries to avoid higher cpu memory consumption
  // happens when the grid config is larger
  const GeometryView& getActiveGeometry(size_t idx) const { return m_geometryViews[idx % m_originalGeometryCount]; }
  size_t              getActiveGeometryCount() const { return m_activeGeometryCount; }

  uint32_t getGeometryInstanceFactor() const
  {
    return m_gridConfig.uniqueGeometriesForCopies ? 1u : uint32_t(m_instances.size() / m_originalInstanceCount);
  }

  bool m_isBig = false;

  uint32_t m_maxClusterTriangles       = 0;
  uint32_t m_maxClusterVertices        = 0;
  uint32_t m_maxPerGeometryClusters    = 0;
  uint32_t m_maxPerGeometryTriangles   = 0;
  uint32_t m_maxPerGeometryVertices    = 0;
  uint32_t m_maxLodLevelsCount         = 0;
  uint32_t m_hiPerGeometryClusters     = 0;
  uint32_t m_hiPerGeometryTriangles    = 0;
  uint32_t m_hiPerGeometryVertices     = 0;
  uint64_t m_hiClustersCount           = 0;
  uint64_t m_hiTrianglesCount          = 0;
  uint64_t m_hiClustersCountInstanced  = 0;
  uint64_t m_hiTrianglesCountInstanced = 0;
  uint64_t m_totalClustersCount        = 0;
  uint64_t m_totalTrianglesCount       = 0;
  uint64_t m_totalVerticesCount        = 0;

  std::vector<uint32_t> m_clusterTriangleHistogram;
  std::vector<uint32_t> m_clusterVertexHistogram;
  std::vector<uint32_t> m_groupClusterHistogram;
  std::vector<uint32_t> m_nodeChildrenHistogram;
  std::vector<uint32_t> m_lodLevelsHistogram;

  uint32_t m_clusterTriangleHistogramMax;
  uint32_t m_clusterVertexHistogramMax;
  uint32_t m_groupClusterHistogramMax;
  uint32_t m_nodeChildrenHistogramMax;
  uint32_t m_lodLevelsHistogramMax;

  bool m_loadedFromCache   = false;
  bool m_hasVertexNormals  = false;
  bool m_hasVertexUVs      = false;
  bool m_hasVertexTangents = false;

  size_t m_originalInstanceCount = 0;
  size_t m_originalGeometryCount = 0;

  size_t m_cacheFileSize = 0;

private:
  static bool     loadCached(GeometryView& view, uint64_t dataSize, const void* data);
  static bool     storeCached(const GeometryView& view, uint64_t dataSize, void* data);
  static uint64_t storeCached(const GeometryView& view, FILE* outFile);

  // GeometryStorage allows building and modifying the data in system RAM
  struct GeometryStorage : GeometryBase
  {
    // temporary, removed after processing
    std::vector<glm::vec3>  vertexPositions;
    std::vector<float>      vertexAttributes;
    std::vector<glm::uvec3> triangles;

    uint32_t attributesWithWeights  = 0u;
    uint32_t attributeNormalOffset  = ~0u;
    uint32_t attributeUvOffset      = ~0u;
    uint32_t attributeTangentOffset = ~0u;

    // persistent used in view
    std::vector<uint8_t>   groupData;
    std::vector<GroupInfo> groupInfos;

    std::vector<shaderio::LodLevel> lodLevels;
    std::vector<shaderio::BBox>     lodNodeBboxes;
    std::vector<shaderio::Node>     lodNodes;

    std::vector<uint32_t> localMaterialIDs;
  };

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
      uint64_t magic            = 0x006f65676e73766eULL;  // nvsngeo
      uint32_t geoVersion       = 6;
      uint32_t geoStructSize    = uint32_t(sizeof(GeometryView));
      uint32_t configVersion    = SceneConfig::version;
      uint32_t configStructSize = uint32_t(sizeof(SceneConfig));
      uint64_t alignment        = nvclusterlod::detail::ALIGNMENT;

      // geoVersion history:
      // 1 initial
      // 2 bugfix wrong storage of `lodInfo`
      // 3 octant vertices
      // 4 table is 2 x 64-bit per geometry (offset + size) to allow out of order storage
      // 5
      // 6 reduced shaderio::Group/Cluster structs using relative offsets
    };

    Header header;

  public:
    SceneConfig config;
  };

  static_assert(sizeof(CacheFileHeader) % nvclusterlod::detail::ALIGNMENT == 0, "CacheFileHeader size unaligned");

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

    void getSceneLodSettings(SceneConfig& settings) const;

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


  size_t m_activeGeometryCount = 0;

  std::vector<GeometryStorage> m_geometryStorages;
  std::vector<GeometryView>    m_geometryViews;

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

  // only used in `processingOnly` mode
  FILE*                 m_processingOnlyFile             = nullptr;
  FILE*                 m_processingOnlyPartialFile      = nullptr;
  size_t                m_processingOnlyPartialCompleted = 0;
  uint64_t              m_processingOnlyFileOffset       = 0;
  std::vector<uint64_t> m_processingOnlyGeometryOffsets;

  struct ProcessingInfo
  {
    // how we perform multi-threading:
    // - either over geometries (outer loop)
    // - or within a geometry (inner loops)

    nvcluster_Context    clusterContext{};
    nvclusterlod_Context lodContext{};

    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;
    uint32_t numInnerThreads = 1;

    size_t geometryCount = 0;

    std::mutex processOnlySaveMutex;

    // logging progress

    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    std::mutex progressMutex;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;

    void init(float pct);
    // parallelismMode: <0 inner, ==0 auto, >0 outer
    void setupParallelism(size_t geometryCount_, size_t geometryCompletedCount, int parallelismMode);
    void deinit();

    void     logBegin();
    uint32_t logCompletedGeometry();
    void     logEnd();
  };

  Result loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath);

  void openCache();
  void closeCache();

  bool checkCache(const GeometryLodInput& info, size_t geometryIndex);

  void processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached);
  void loadCachedGeometry(GeometryStorage& geometry, size_t geometryIndex);

  void buildGeometryClusterLod(const ProcessingInfo& processingInfo, GeometryStorage& geometry);
  void buildGeometryClusterLodNvLib(const ProcessingInfo& processingInfo, GeometryStorage& geometry);
  void buildGeometryClusterLodMeshoptimizer(const ProcessingInfo& processingInfo, GeometryStorage& geometry);

  void computeLodBboxes_recursive(GeometryStorage& geometry, size_t nodeIdx);
  void buildGeometryDedupVertices(const ProcessingInfo& processingInfo, GeometryStorage& geometry);

  void beginProcessingOnly(size_t geometryCount);
  void saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex);
  bool endProcessingOnly(bool hadError);

  void computeClusterStats();
  void computeHistograms();
  void computeInstanceBBoxes();

  struct TempContext
  {
    const ProcessingInfo& processingInfo;
    GeometryStorage&      geometry;

    bool      innerThreadingActive   = false;
    bool      levelGroupOffsetValid  = false;
    GroupInfo threadGroupInfo        = {};
    uint32_t  threadGroupSize        = 0;
    uint32_t  threadGroupStorageSize = 0;
    uint32_t  lodLevel               = ~0u;
    size_t    levelGroupOffset       = 0;


    std::mutex           groupMutex;
    std::atomic_uint32_t groupIndexOrdered = 0;
    std::atomic_size_t   groupDataOrdered  = 0;
    std::vector<uint8_t> threadGroupDatas;

    // only for nvlib
    nvclusterlod::LodMesh      lodMesh;
    nvclusterlod::LodHierarchy lodHierarchy;
    std::vector<uint8_t>       groupLodLevels;
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

  static uint32_t storeGroup(TempContext*                         context,
                             uint32_t                             threadIndex,
                             uint32_t                             groupIndex,
                             const TempGroup&                     tempGroup,
                             std::function<TempCluster(uint32_t)> tempClusterFn);

  void buildGeometryLodHierarchyMeshoptimizer(const ProcessingInfo& processingInfo, GeometryStorage& geometry);


  static void clodIterationMeshoptimizer(void* iteration_context, void* output_context, int depth, size_t task_count);
  static int  clodGroupMeshoptimizer(void*              output_context,
                                     clodGroup          group,
                                     const clodCluster* clusters,
                                     size_t             cluster_count,
                                     size_t             task_index,
                                     uint32_t           thread_index);
};

}  // namespace lodclusters
