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

#include <glm/glm.hpp>
#include <nvclusterlod/nvclusterlod_hierarchy_storage.hpp>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>

#include "shaders/shaderio_scene.h"


namespace lodclusters {

enum ClusterBuilderType
{
  CLUSTER_BUILDER_NVCLUSTER,
  CLUSTER_BUILDER_FILE,
};

struct SceneConfig
{
  ClusterBuilderType clusterBuilderType       = CLUSTER_BUILDER_NVCLUSTER;
  uint32_t           clusterVertices          = 64;
  uint32_t           clusterTriangles         = 64;
  uint32_t           clusterGroupSize         = 32;
  bool               clusterStripify          = true;
  float              lodLevelDecimationFactor = 0.5f;
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

  struct Geometry
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

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;

    std::vector<glm::uvec3> globalTriangles;

    // local to a cluster: indices of triangle vertices and global vertices
    std::vector<uint8_t>  localTriangles;
    std::vector<uint32_t> localVertices;

    std::vector<nvcluster::Range> clusterVertexRanges;
    std::vector<shaderio::BBox>   clusterBboxes;
    std::vector<uint8_t>          groupLodLevels;

    nvclusterlod::LodMesh       lodMesh;
    nvclusterlod::LodHierarchy  lodHierachy;
    std::vector<shaderio::BBox> nodeBboxes;

    // for streaming
    nvclusterlod::GroupGeneratingGroups groupGeneratingGroups;
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
  void deinit();

  void updateSceneGrid(const SceneGridConfig& gridConfig);

  SceneConfig m_config;

  shaderio::BBox m_bbox;

  std::vector<Instance> m_instances;
  std::vector<Camera>   m_cameras;

  // we virtually instance geometries to avoid cpu memory consumption
  // happens when the grid config is larger
  const Geometry& getActiveGeometry(size_t idx) const { return m_geometries[idx % m_originalGeometryCount]; }
  size_t          getActiveGeometryCount() const { return m_activeGeometryCount; }

  uint32_t              m_maxPerGeometryClusters   = 0;
  uint32_t              m_maxPerGeometryTriangles  = 0;
  uint32_t              m_maxPerGeometryVertices   = 0;
  uint32_t              m_hiPerGeometryClusters    = 0;
  uint32_t              m_hiPerGeometryTriangles   = 0;
  uint32_t              m_hiPerGeometryVertices    = 0;
  uint64_t              m_hiClustersCount          = 0;
  uint64_t              m_hiTrianglesCount         = 0;
  uint64_t              m_totalClustersCount       = 0;
  uint32_t              m_clusterMaxVerticesCount  = 0;
  uint32_t              m_clusterMaxTrianglesCount = 0;
  std::vector<uint32_t> m_clusterTriangleHistogram;
  std::vector<uint32_t> m_clusterVertexHistogram;
  std::vector<uint32_t> m_groupClusterHistogram;
  std::vector<uint32_t> m_nodeChildrenHistogram;

  uint32_t m_clusterTriangleHistogramMax;
  uint32_t m_clusterVertexHistogramMax;
  uint32_t m_groupClusterHistogramMax;
  uint32_t m_nodeChildrenHistogramMax;

private:
  size_t m_originalInstanceCount = 0;
  size_t m_originalGeometryCount = 0;

  size_t m_activeGeometryCount = 0;

  std::vector<Geometry> m_geometries;

  bool loadGLTF(const char* filename);

  void buildGeometryClusters(nvclusterlod::Context lodcontext, Geometry& geometry);
  void computeLodBboxes_recursive(Geometry& geom, size_t nodeIdx);
  void buildGeometryBboxes(Geometry& geometry);
  void buildGeometryClusterStrips(Geometry& geom, uint64_t& totalTriangles, uint64_t& totalStrips);
  void buildGeometryClusterVertices(Geometry& geometry);
  void buildClusters();
  void computeHistograms();
  void computeInstanceBBoxes();
};

}  // namespace lodclusters
