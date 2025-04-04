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

#include <glm/glm.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy_storage.hpp>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <nvclusterlod/nvclusterlod_cache.hpp>

#include "shaders/shaderio_scene.h"

namespace lodclusters {

struct SceneConfig
{
  uint32_t clusterVertices          = 64;
  uint32_t clusterTriangles         = 64;
  uint32_t clusterGroupSize         = 32;
  bool     clusterStripify          = true;
  float    lodLevelDecimationFactor = 0.5f;
  bool     autoSaveCache            = false;
  bool     autoLoadCache            = true;

  // TODO avoid GeometryStorage completely when using mapping cache file
  //bool usePersistentMapping     = false;
};

struct SceneGridConfig
{
  // when set to true each new set of instance on the grid gets
  // its own unique set of geometries. This stresses the streaming system
  // and memory consumption a lot more.
  bool      uniqueGeometriesForCopies = true;
  uint32_t  numCopies                 = 1;
  uint32_t  gridBits                  = 13;
  glm::vec3 refShift                  = {1.0f, 1.0f, 1.0f};
  float     snapAngle                 = 0;
};

class Scene
{
public:
  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
  };

  struct GeometryBase
  {
    uint32_t clusterMaxVerticesCount;
    uint32_t clusterMaxTrianglesCount;

    uint32_t lodLevelsCount;

    // based on highest detail lod
    uint32_t hiTriangleCount;
    uint32_t hiVerticesCount;
    uint32_t hiClustersCount;

    // total sum
    uint32_t totalTriangleCount;
    uint32_t totalVerticesCount;
    uint32_t totalClustersCount;

    shaderio::BBox bbox;

    nvclusterlod::LodGeometryInfo lodInfo;
  };

  // not fully leveraged yet, but preparation to have a readonly view on
  // data accessed from persistently memory mapped files
  struct GeometryView : GeometryBase
  {
    std::span<const glm::vec3> positions;
    std::span<const glm::vec3> normals;

    std::span<const uint8_t> localTriangles;

    std::span<const nvcluster::Range> clusterVertexRanges;
    std::span<const shaderio::BBox>   clusterBboxes;
    std::span<const uint8_t>          groupLodLevels;

    std::span<const shaderio::BBox> nodeBboxes;

    nvclusterlod::LodMeshView      lodMesh;
    nvclusterlod::LodHierarchyView lodHierarchy;

    inline uint64_t getCachedSize() const
    {
      uint64_t cachedSize = 0;

      cachedSize += (sizeof(GeometryBase) + nvclusterlod::detail::ALIGN_MASK) & ~nvclusterlod::detail::ALIGN_MASK;
      cachedSize += nvclusterlod::detail::getCachedSize(positions);
      cachedSize += nvclusterlod::detail::getCachedSize(normals);
      cachedSize += nvclusterlod::detail::getCachedSize(localTriangles);
      cachedSize += nvclusterlod::detail::getCachedSize(clusterVertexRanges);
      cachedSize += nvclusterlod::detail::getCachedSize(clusterBboxes);
      cachedSize += nvclusterlod::detail::getCachedSize(groupLodLevels);
      cachedSize += nvclusterlod::detail::getCachedSize(nodeBboxes);
      cachedSize += nvclusterlod::getCachedSize(lodMesh);
      cachedSize += nvclusterlod::getCachedSize(lodHierarchy);

      return cachedSize;
    }
  };

  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };

  bool init(const char* filename, const SceneConfig& config);
  bool saveCache() const;
  void deinit();

  void updateSceneGrid(const SceneGridConfig& gridConfig);

  SceneConfig m_config;

  shaderio::BBox m_bbox;

  std::vector<Instance> m_instances;
  std::vector<Camera>   m_cameras;

  // we virtually instance geometries to avoid cpu memory consumption
  // happens when the grid config is larger
  const GeometryView& getActiveGeometry(size_t idx) const { return m_geometryViews[idx % m_originalGeometryCount]; }
  size_t              getActiveGeometryCount() const { return m_activeGeometryCount; }

  uint32_t              m_maxPerGeometryClusters    = 0;
  uint32_t              m_maxPerGeometryTriangles   = 0;
  uint32_t              m_maxPerGeometryVertices    = 0;
  uint32_t              m_hiPerGeometryClusters     = 0;
  uint32_t              m_hiPerGeometryTriangles    = 0;
  uint32_t              m_hiPerGeometryVertices     = 0;
  uint64_t              m_hiClustersCount           = 0;
  uint64_t              m_hiTrianglesCount          = 0;
  uint64_t              m_hiClustersCountInstanced  = 0;
  uint64_t              m_hiTrianglesCountInstanced = 0;
  uint64_t              m_totalClustersCount        = 0;
  uint32_t              m_clusterMaxVerticesCount   = 0;
  uint32_t              m_clusterMaxTrianglesCount  = 0;
  std::vector<uint32_t> m_clusterTriangleHistogram;
  std::vector<uint32_t> m_clusterVertexHistogram;
  std::vector<uint32_t> m_groupClusterHistogram;
  std::vector<uint32_t> m_nodeChildrenHistogram;

  uint32_t m_clusterTriangleHistogramMax;
  uint32_t m_clusterVertexHistogramMax;
  uint32_t m_groupClusterHistogramMax;
  uint32_t m_nodeChildrenHistogramMax;

  bool m_loadedCache = false;

private:
  static bool loadCached(GeometryView& view, uint64_t dataSize, const void* data);
  static bool storeCached(const GeometryView& view, uint64_t dataSize, void* data);

  // GeometryStorage allows building and modifying the data in RAM
  struct GeometryStorage : GeometryBase
  {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;

    // temporary, removed after processing
    std::vector<glm::uvec3> globalTriangles;

    // local to a cluster: indices of triangle vertices and global vertices
    std::vector<uint8_t> localTriangles;

    // temporary, removed after processing
    std::vector<uint32_t> localVertices;

    std::vector<nvcluster::Range> clusterVertexRanges;
    std::vector<shaderio::BBox>   clusterBboxes;
    std::vector<uint8_t>          groupLodLevels;

    nvclusterlod::LodMesh       lodMesh;
    nvclusterlod::LodHierarchy  lodHierarchy;
    std::vector<shaderio::BBox> nodeBboxes;
  };

  class CacheHeader
  {
  public:
    CacheHeader()
    {
      header = {};
      if(sizeof(CacheHeader) - sizeof(Header))
      {
        memset((&header) + 1, 0, sizeof(CacheHeader) - sizeof(Header));
      }
    }

  private:
    struct Header
    {
      uint64_t magic          = 0x006f65676e73766eULL;  // nvsngeo
      uint32_t geoVersion     = 1;
      uint32_t structSize     = uint32_t(sizeof(GeometryView));
      uint32_t lodVersion     = NVCLUSTERLOD_VERSION;
      uint32_t clusterVersion = NVCLUSTER_VERSION;
    };

    union
    {
      Header  header;
      uint8_t data[(sizeof(Header) + nvclusterlod::detail::ALIGN_MASK) & ~(nvclusterlod::detail::ALIGN_MASK)];
    };
  };

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
      // offsets where each `LodGeometry` data is stored.
      // ordered with ascending offsets
      // `geometryDataSize = geometryOffsets[geometryIndex + 1] - geometryOffsets[geometryIndex];`
      uint64_t geometryOffsets[geometryCount + 1];
      uint64_t geometryCount;
    };
#endif

  public:
    bool isValid() const { return m_dataSize != 0; }

    bool init(uint64_t dataSize, const void* data);

    void deinit() { *(this) = {}; }

    uint64_t getGeometryCount() const { return m_geometryCount; }

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

  size_t m_originalInstanceCount = 0;
  size_t m_originalGeometryCount = 0;

  size_t m_activeGeometryCount = 0;

  std::vector<GeometryStorage> m_geometryStorages;
  std::vector<GeometryView>    m_geometryViews;

  std::string m_filename;

  bool loadGLTF(const char* filename, const CacheFileView& cacheFileView);

  nvcluster::Config getClusterConfig() const
  {
    nvcluster::Config clusterConfig = {};
    clusterConfig.minClusterSize    = (m_config.clusterTriangles * 3) / 4;
    clusterConfig.maxClusterSize    = m_config.clusterTriangles;
    return clusterConfig;
  }
  nvcluster::Config getGroupConfig() const
  {
    nvcluster::Config groupConfig = {};
    groupConfig.minClusterSize    = (m_config.clusterGroupSize * 3) / 4;
    groupConfig.maxClusterSize    = m_config.clusterGroupSize;
    return groupConfig;
  }

  bool checkCache(const nvclusterlod::LodGeometryInfo& info, const CacheFileView& cacheFileView, size_t geometryIndex);

  void loadCachedGeometry(GeometryStorage& geom, const CacheFileView& cacheFileView, size_t geometryIndex);

  void buildGeometryClusters(nvclusterlod::Context lodcontext, GeometryStorage& geometry, uint32_t numThreads);
  void computeLodBboxes_recursive(GeometryStorage& geom, size_t nodeIdx);
  void buildGeometryBboxes(GeometryStorage& geometry, uint32_t numThreads);
  void buildGeometryClusterStrips(GeometryStorage& geom, uint64_t& totalTriangles, uint64_t& totalStrips, uint32_t numThreads);
  void buildGeometryClusterVertices(GeometryStorage& geometry, uint32_t numThreads);
  void buildClusters(const CacheFileView& cacheFileView);

  void computeHistograms();
  void computeInstanceBBoxes();
};

}  // namespace lodclusters
