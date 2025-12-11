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
#include <array>
#include <string>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <functional>

#include <glm/glm.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/alignment.hpp>

#include "serialization.hpp"
#include "meshopt_clusterlod.h"
#include "../shaders/shaderio_scene.h"

namespace lodclusters {

// Controls the scene's data generation during loading and processing.
struct SceneConfig
{
  static const uint32_t version = 2;

  // cluster and cluster group settings
  uint32_t clusterVertices    = 128;
  uint32_t clusterTriangles   = 128;
  uint32_t clusterGroupSize   = 32;
  uint32_t preferredNodeWidth = 8;

  // default setting should prefer ray tracing
  bool meshoptPreferRayTracing = true;

  // store groups in a compressed way
  // uncompress at runtime
  bool useCompressedData = false;

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
  float simplifyTangentWeight     = 0.01f;
  float simplifyTangentSignWeight = 0.5f;
  float simplifyTexCoordWeight    = 0;

  // used when compression is enabled
  uint32_t compressionPosDropBits = 7;
  uint32_t compressionTexDropBits = 7;

  // experimental meshoptimizer, try to remove small triangles despite high error
  float lodErrorEdgeLimit = 0;

  // want to allow some binary compatibility with older cache files
  // safe to add new variables into this section as long as they are zeroed by default
  uint32_t reservedData[14] = {};
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

  // optional thread-safe progress bar updates
  std::atomic_uint32_t* progressPct = nullptr;
};

// To artificially instance the full scene on a grid multiple times.
// Useful for benchmarking.
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

  Result init(const std::filesystem::path& filePath, const SceneConfig& config, const SceneLoaderConfig& loaderConfig, bool skipCache);
  bool saveCache() const;
  void deinit();

  void updateSceneGrid(const SceneGridConfig& gridConfig);

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
  //
  struct GroupInfo
  {
    uint64_t offsetBytes : 42;
    uint64_t sizeBytes : 22;
    uint16_t vertexCount;
    uint16_t triangleCount;
    uint8_t  lodLevel;
    uint8_t  clusterCount;
    uint8_t  attributeBits;
    uint8_t  reserved1 = 0;
    uint64_t vertexDataCount : 21;
    // these must be 0 if group is stored 'uncompressed'
    // otherwise they provide the size information of the uncompressed state.
    uint64_t uncompressedVertexDataCount : 21;
    uint64_t uncompressedSizeBytes : 22;

    // compression may impact the size on device
    uint32_t getDeviceSize() const { return uint32_t(uncompressedSizeBytes ? uncompressedSizeBytes : sizeBytes); }

    // safe upper bound
    uint32_t estimateVertexDataCount() const
    {
      uint32_t dataCount = vertexCount * 3;
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
      {
        dataCount += vertexCount * 1;
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
      {
        dataCount += vertexCount * 2;
        dataCount += clusterCount;  // potential padding
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1)
      {
        dataCount += vertexCount * 2;
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
      threadGroupSize        = threadGroupSize + sizeof(uint8_t) * triangleCount * 3;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 8) + sizeof(float) * vertexDataCount;
      return nvutils::align_up(threadGroupSize, 16);
    }

    size_t computeUncompressedSectionSize() const
    {
      size_t threadGroupSize = sizeof(shaderio::Group);
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::Cluster) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 4) + sizeof(uint32_t) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::BBox) * clusterCount;
      threadGroupSize        = threadGroupSize + sizeof(uint8_t) * triangleCount * 3;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 8);
      return threadGroupSize;
    }
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
    std::span<const uint8_t>           indices;
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

      indices = std::span((const uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleCount * 3);

      vertices = std::span((const float*)nvutils::align_up(size_t(indices.data() + info.triangleCount * 3), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }

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

  // read-write accessor used for processing a cluster group.
  // same structure as above
  struct GroupStorage
  {
    uint8_t*                     raw;
    const size_t                 rawSize = 0;
    shaderio::Group*             group;
    std::span<shaderio::Cluster> clusters;
    std::span<uint32_t>          clusterGeneratingGroups;
    std::span<shaderio::BBox>    clusterBboxes;
    std::span<uint8_t>           indices;
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
      indices = std::span((uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleCount * 3);
      vertices = std::span((float*)nvutils::align_up(size_t(indices.data() + info.triangleCount * 3), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }

    // cluster data pointers are stored as offsets relative to the Cluster's header.
    uint32_t getClusterLocalOffset(uint32_t clusterIndex, const void* input, size_t overrideSize = 0) const
    {
      assert(size_t(input) >= size_t(&clusters[clusterIndex]));
      assert(size_t(input) < size_t(raw + (overrideSize ? overrideSize : rawSize)));

      return uint32_t(size_t(input) - size_t(&clusters[clusterIndex]));
    }

    uint32_t* getClusterVertexData(uint32_t clusterIndex)
    {
      return (uint32_t*)(size_t(&clusters[clusterIndex]) + clusters[clusterIndex].vertices);
    }
  };


  // used for preloaded groups, streamed in groups are patched in shaders.
  static void fillGroupRuntimeData(const GroupInfo& srcGroupInfo,
                                   const GroupView& srcGroupView,
                                   uint32_t         groupID,
                                   uint32_t         groupResidentID,
                                   uint32_t         clusterResidentID,
                                   void*            dst,
                                   size_t           dstSize);

  // used to decompress group on CPU.
  static void decompressGroup(const GroupInfo& info, const GroupView& groupView, void* dst, size_t dstSize);


  //////////////////////////////////////////////////////////////////////////

  // Geometry

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

  // read-only accessor for the geometry data.
  // used at runtime.
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

  // we virtually instance geometries to avoid higher cpu memory consumption
  // happens when the grid config is larger
  const GeometryView& getActiveGeometry(size_t idx) const { return m_geometryViews[idx % m_originalGeometryCount]; }
  size_t              getActiveGeometryCount() const { return m_activeGeometryCount; }

  uint32_t getGeometryInstanceFactor() const
  {
    return m_gridConfig.uniqueGeometriesForCopies ? 1u : uint32_t(m_instances.size() / m_originalInstanceCount);
  }


  //////////////////////////////////////////////////////////////////////////

  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
    uint32_t       materialID = ~0U;
    bool           twoSided   = false;
    glm::vec4      color{0.8, 0.8, 0.8, 1.0f};
  };

  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };

  //////////////////////////////////////////////////////////////////////////

  // statistics

  struct Histograms
  {
    static const uint32_t version = 1;

    std::array<uint32_t, 256 + 1>                         clusterTriangles = {};
    std::array<uint32_t, 256 + 1>                         clusterVertices  = {};
    std::array<uint32_t, SHADERIO_MAX_GROUP_CLUSTERS + 1> groupClusters    = {};
    std::array<uint32_t, SHADERIO_MAX_NODE_CHILDREN + 1>  nodeChildren     = {};
    std::array<uint32_t, SHADERIO_MAX_LOD_LEVELS + 1>     lodLevels        = {};

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

  shaderio::BBox m_bbox;
  shaderio::BBox m_gridBbox;

  std::vector<Instance> m_instances;
  std::vector<Camera>   m_cameras;

  bool m_isBig       = false;
  bool m_hasTwoSided = false;

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

    uint32_t attributesWithWeights  = 0u;
    uint32_t attributeNormalOffset  = ~0u;
    uint32_t attributeTex0offset    = ~0u;
    uint32_t attributeTex1offset    = ~0u;
    uint32_t attributeTangentOffset = ~0u;

    // persistent used in view
    std::vector<uint8_t>   groupData;
    std::vector<GroupInfo> groupInfos;

    std::vector<shaderio::LodLevel> lodLevels;
    std::vector<shaderio::BBox>     lodNodeBboxes;
    std::vector<shaderio::Node>     lodNodes;

    std::vector<uint32_t> localMaterialIDs;
  };

  size_t m_activeGeometryCount = 0;

  std::vector<GeometryStorage> m_geometryStorages;
  std::vector<GeometryView>    m_geometryViews;

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
      uint32_t geoVersion          = 7;
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

    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;
    uint32_t numInnerThreads = 1;

    // if triangleCount is not 0, then we will track progress
    // based on completed triangles, otherwise based on
    // completed geometries
    size_t   geometryCount = 0;
    uint64_t triangleCount = 0;

    std::mutex processOnlySaveMutex;

    // bufferview compression

    std::vector<uint32_t> bufferViewUsers;
    std::vector<uint32_t> bufferViewLocks;

    // stats

    struct Stats
    {
      std::atomic_uint64_t groups                = 0;
      std::atomic_uint64_t clusters              = 0;
      std::atomic_uint64_t vertices              = 0;
      std::atomic_uint64_t groupUniqueVertices   = 0;
      std::atomic_uint64_t groupHeaderBytes      = 0;
      std::atomic_uint64_t triangleIndexBytes    = 0;
      std::atomic_uint64_t vertexPosBytes        = 0;
      std::atomic_uint64_t vertexTexCoordBytes   = 0;
      std::atomic_uint64_t vertexNrmBytes        = 0;
      std::atomic_uint64_t vertexCompressedBytes = 0;
      std::atomic_uint64_t clusterBboxBytes      = 0;
      std::atomic_uint64_t clusterHeaderBytes    = 0;
      std::atomic_uint64_t clusterGenBytes       = 0;
    } stats;


    // logging progress

    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    uint64_t   progressTrianglesCompleted  = 0;
    std::mutex progressMutex;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;

    void init(float pct);
    // parallelismMode: <0 inner, ==0 auto, >0 outer
    void setupParallelism(size_t geometryCount_, size_t geometryCompletedCount, int parallelismMode);
    void setupCompressedGltf(size_t bufferViewCount);
    void deinit();

    void     logBegin(uint64_t totalTriangleCount);
    uint32_t logCompletedGeometry(uint64_t triangleCount = 0);
    void     logEnd();
  };

  Result loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath);

private:
  void loadGeometryGLTF(ProcessingInfo& processingInfo, uint64_t geometryIndex, size_t meshIndex, const struct cgltf_data* gltf);
  void addInstancesFromNodeGLTF(const std::vector<size_t>& meshToGeometry,
                                const struct cgltf_data*   data,
                                const struct cgltf_node*   node,
                                const glm::mat4            parentObjToWorldTransform = glm::mat4(1));

  // to handle glTF EXT_meshopt_compression
  bool loadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                               std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                               const struct cgltf_data*                       gltf);
  void unloadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                 std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                                 const struct cgltf_data*                       gltf);

  void processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached);

  void buildGeometryLod(ProcessingInfo& processingInfo, GeometryStorage& geometry);
  void buildGeometryLodHierarchy(ProcessingInfo& processingInfo, GeometryStorage& geometry);

  void computeLodBboxes_recursive(GeometryStorage& geometry, size_t nodeIdx);
  void buildGeometryDedupVertices(ProcessingInfo& processingInfo, GeometryStorage& geometry);

  void computeHistogramMaxs();
  void computeInstanceBBoxes();

  // these modes always output to the cache directly
  void beginProcessingOnly(size_t geometryCount);
  void saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex);
  bool endProcessingOnly(bool hadError);


  //////////////////////////////////////////////////////////////////////////

  // Cluster Group Building

  struct TempContext
  {
    ProcessingInfo&  processingInfo;
    GeometryStorage& geometry;
    Scene&           scene;

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

  uint32_t storeGroup(TempContext*       context,
                      uint32_t           threadIndex,
                      uint32_t           groupIndex,
                      const clodGroup&   group,
                      uint32_t           clusterCount,
                      const clodCluster* clusters);

  void compressGroup(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo, uint32_t* vertexCacheLocal);

  static void clodIterationMeshoptimizer(void* iteration_context, void* output_context, int depth, size_t task_count);
  static int  clodGroupMeshoptimizer(void*              output_context,
                                     clodGroup          group,
                                     const clodCluster* clusters,
                                     size_t             cluster_count,
                                     size_t             task_index,
                                     uint32_t           thread_index);
};

}  // namespace lodclusters
