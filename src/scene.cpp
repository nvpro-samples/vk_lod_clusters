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

#include <random>
#include <string.h>

#include <meshoptimizer.h>
#include <nvh/nvprint.hpp>
#include <nvh/parallel_work.hpp>
#include <nvh/profiler.hpp>
#include <nvh/filemapping.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "scene.hpp"


namespace lodclusters {
static_assert(sizeof(shaderio::Node) == sizeof(nvclusterlod::Node));
static_assert(offsetof(shaderio::Node, nodeRange) == offsetof(nvclusterlod::Node, children));
static_assert(offsetof(shaderio::Node, traversalMetric) == offsetof(nvclusterlod::Node, boundingSphere));
static_assert(offsetof(shaderio::Node, traversalMetric) + offsetof(shaderio::TraversalMetric, maxQuadricError)
              == offsetof(nvclusterlod::Node, maxClusterQuadricError));
static_assert(sizeof(glm::vec4) == sizeof(nvclusterlod::Sphere));
static_assert(nvclusterlod::ORIGINAL_MESH_GROUP == SHADERIO_ORIGINAL_MESH_GROUP);

bool Scene::init(const char* filename, const SceneConfig& config)
{
  *this = {};

  m_filename            = filename;
  m_config              = config;
  m_prebuildLodClusters = false;

  if(!loadGLTF(filename))
  {
    return false;
  }

  computeInstanceBBoxes();

  buildClusters();

  for(auto& geom : m_geometries)
  {
    m_hiPerGeometryTriangles   = std::max(m_hiPerGeometryTriangles, geom.hiTriangleCount);
    m_hiPerGeometryVertices    = std::max(m_hiPerGeometryVertices, geom.hiVerticesCount);
    m_hiPerGeometryClusters    = std::max(m_hiPerGeometryClusters, geom.hiClustersCount);
    m_maxPerGeometryTriangles  = std::max(m_maxPerGeometryTriangles, geom.totalTriangleCount);
    m_maxPerGeometryVertices   = std::max(m_maxPerGeometryVertices, geom.totalVerticesCount);
    m_maxPerGeometryClusters   = std::max(m_maxPerGeometryClusters, geom.totalClustersCount);
    m_clusterMaxVerticesCount  = std::max(m_clusterMaxVerticesCount, geom.clusterMaxVerticesCount);
    m_clusterMaxTrianglesCount = std::max(m_clusterMaxTrianglesCount, geom.clusterMaxTrianglesCount);
    m_hiTrianglesCount += geom.hiTriangleCount;
    m_hiClustersCount += geom.hiClustersCount;
    m_totalClustersCount += geom.totalClustersCount;
  }
  for(size_t i = 0; i < m_instances.size(); i++)
  {
    const Geometry& geom = m_geometries[m_instances[i].geometryID];
    m_hiTrianglesCountInstanced += geom.hiTriangleCount;
    m_hiClustersCountInstanced += geom.hiClustersCount;
  }


  m_originalInstanceCount = m_instances.size();
  m_originalGeometryCount = m_geometries.size();
  m_activeGeometryCount   = m_originalGeometryCount;

  if(!m_prebuildLodClusters && m_config.autoSaveCache)
  {
    saveCache();
  }

  return true;
}

void Scene::deinit()
{
  *this = {};
}

void Scene::updateSceneGrid(const SceneGridConfig& gridConfig)
{
  size_t copiesCount = std::max(1u, gridConfig.numCopies);

  size_t numOldCopies = m_instances.size() / m_originalInstanceCount;

  m_instances.resize(m_originalInstanceCount * copiesCount);
  m_activeGeometryCount = gridConfig.uniqueGeometriesForCopies ? m_originalGeometryCount * copiesCount : m_originalGeometryCount;

  std::default_random_engine            rng(2342);
  std::uniform_real_distribution<float> randomUnorm(0.0f, 1.0f);

  uint32_t axis    = gridConfig.gridBits;
  size_t   sq      = 1;
  int      numAxis = 0;
  if(!axis)
    axis = 3;

  for(int i = 0; i < 3; i++)
  {
    numAxis += (axis & (1 << i)) ? 1 : 0;
  }

  switch(numAxis)
  {
    case 1:
      sq = copiesCount;
      break;
    case 2:
      while(sq * sq < copiesCount)
      {
        sq++;
      }
      break;
    case 3:
      while(sq * sq * sq < copiesCount)
      {
        sq++;
      }
      break;
  }


  size_t lastCopyIndex = 0;

  glm::vec3 gridShift;
  glm::mat4 gridRotMatrix;

  for(size_t copyIndex = 1; copyIndex < copiesCount; copyIndex++)
  {
    gridShift = gridConfig.refShift * (m_bbox.hi - m_bbox.lo);
    size_t c  = copyIndex;

    float u = 0;
    float v = 0;
    float w = 0;

    switch(numAxis)
    {
      case 1:
        u = float(c);
        break;
      case 2:
        u = float(c % sq);
        v = float(c / sq);
        break;
      case 3:
        u = float(c % sq);
        v = float((c / sq) % sq);
        w = float(c / (sq * sq));
        break;
    }

    float use = u;

    if(axis & (1 << 0))
    {
      gridShift.x *= -use;
      if(numAxis > 1)
        use = v;
    }
    else
    {
      gridShift.x = 0;
    }

    if(axis & (1 << 1))
    {
      gridShift.y *= use;
      if(numAxis > 2)
        use = w;
      else if(numAxis > 1)
        use = v;
    }
    else
    {
      gridShift.y = 0;
    }

    if(axis & (1 << 2))
    {
      gridShift.z *= -use;
    }
    else
    {
      gridShift.z = 0;
    }

    if(axis & (8 | 16 | 32))
    {
      glm::vec3 mask    = {axis & 8 ? 1.0f : 0.0f, axis & 16 ? 1.0f : 0.0f, axis & 32 ? 1.0f : 0.0f};
      glm::vec3 gridDir = glm::vec3(randomUnorm(rng), randomUnorm(rng), randomUnorm(rng));
      gridDir           = glm::max(gridDir * mask, mask * 0.00001f);
      float gridAngle   = randomUnorm(rng) * 360.0f;
      gridDir           = glm::normalize(gridDir);

      // snap angle
      if(gridConfig.snapAngle > 0.0)
      {
        float remainder = std::fmod(gridAngle, gridConfig.snapAngle);
        gridAngle       = gridAngle - remainder;
      }

      // to radians
      gridAngle = gridAngle * glm::pi<float>() / 180.0f;

      gridRotMatrix = glm::rotate(glm::mat4(1), gridAngle, gridDir);
    }

    for(size_t i = 0; i < m_originalInstanceCount; i++)
    {
      Instance& instance = m_instances[i + copyIndex * m_originalInstanceCount];
      // copy state from reference
      instance = m_instances[i];

      if(gridConfig.uniqueGeometriesForCopies)
      {
        // apply unique set if geometryIDs to this set of instances
        instance.geometryID += uint32_t(c * m_originalGeometryCount);
      }

      // modify matrix for the grid
      glm::mat4 worldMatrix = m_instances[i].matrix;


      glm::vec3 translation;
      translation = worldMatrix[3];
      if(axis & (8 | 16 | 32))
      {
        worldMatrix[3] = glm::vec4(0, 0, 0, 1);
        worldMatrix    = gridRotMatrix * worldMatrix;
      }
      worldMatrix[3] = glm::vec4(translation + gridShift, 1.f);

      instance.matrix = worldMatrix;
    }
  }
}

void Scene::computeInstanceBBoxes()
{
  m_bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  for(auto& instance : m_instances)
  {
    assert(instance.geometryID <= m_geometries.size());

    const Geometry& geom = m_geometries[instance.geometryID];

    instance.bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

    for(uint32_t v = 0; v < 8; v++)
    {
      bool x = (v & 1) != 0;
      bool y = (v & 2) != 0;
      bool z = (v & 4) != 0;

      glm::bvec3 weight(x, y, z);
      glm::vec3  corner = glm::mix(geom.bbox.lo, geom.bbox.hi, weight);
      corner            = instance.matrix * glm::vec4(corner, 1.0f);
      instance.bbox.lo  = glm::min(instance.bbox.lo, corner);
      instance.bbox.hi  = glm::max(instance.bbox.hi, corner);
    }

    m_bbox.lo = glm::min(m_bbox.lo, instance.bbox.lo);
    m_bbox.hi = glm::max(m_bbox.hi, instance.bbox.hi);
  }
}

bool Scene::saveCache()
{
  uint64_t dataSize = sizeof(nvclusterlod::CacheHeader);

  std::vector<uint64_t> geometryOffsets;
  geometryOffsets.reserve(m_geometries.size() + 2);

  for(const Geometry& geom : m_geometries)
  {
    nvclusterlod::LodGeometryView view;
    view.info = geom.lodInfo;
    nvclusterlod::toView(geom.lodMesh, view.lodMesh);
    nvclusterlod::toView(geom.lodHierachy, view.lodHierarchy);

    geometryOffsets.push_back(dataSize);

    dataSize += nvclusterlod::getCachedSize(view);
  }
  geometryOffsets.push_back(dataSize);
  geometryOffsets.push_back(m_geometries.size());

  dataSize += geometryOffsets.size() * sizeof(uint64_t);

  nvh::FileReadOverWriteMapping outMapping;

  std::string outFilename = m_filename + ".nvcllod";

  if(!outMapping.open(outFilename.c_str(), dataSize))
  {
    LOGE("Scene::storeClusterLod failed to save file %s", outFilename.c_str());
    return false;
  }

  uint8_t* mappingData = static_cast<uint8_t*>(outMapping.data());

  // write header
  nvclusterlod::CacheHeader cacheHeader;
  memcpy(mappingData, &cacheHeader, sizeof(cacheHeader));
  // write offset table at end
  memcpy(mappingData + geometryOffsets[m_geometries.size()], geometryOffsets.data(), sizeof(uint64_t) * geometryOffsets.size());

  bool hadError = false;
  nvh::parallel_batches(m_geometries.size(), [&](uint64_t idx) {
    Geometry& geom = m_geometries[idx];

    nvclusterlod::LodGeometryView view;
    view.info = geom.lodInfo;
    nvclusterlod::toView(geom.lodMesh, view.lodMesh);
    nvclusterlod::toView(geom.lodHierachy, view.lodHierarchy);

    uint64_t dataOffset = geometryOffsets[idx];
    uint64_t dataSize   = geometryOffsets[idx + 1] - dataOffset;

    if(!nvclusterlod::storeCached(view, dataSize, mappingData + dataOffset))
    {
      hadError = true;
    }
  });

  if(hadError)
  {
    LOGE("Scene::storeClusterLod had errors while saving %s", outFilename.c_str());
  }
  else
  {
    LOGI("Scene::storeClusterLod saved %s", outFilename.c_str());
  }

  return !hadError;
}

void Scene::buildClusters()
{
  nvh::Profiler::Clock clock;
  double               startTime = clock.getMicroSeconds();

  // can we find a cache file, if so memory map it
  nvh::FileReadMapping    readMapping;
  nvclusterlod::CacheView cacheView;

  if(readMapping.open((m_filename + ".nvcllod").c_str()) && false)
  {
    cacheView.init(readMapping.size(), readMapping.data());
    if(cacheView.isValid() && cacheView.getGeometryCount() != m_geometries.size())
    {
      cacheView.deinit();
    }
    else
    {
      m_prebuildLodClusters = true;
    }
  }

  nvcluster::ContextCreateInfo clusterContextInfo;
  nvcluster::Context           clusterContext;
  nvclusterCreateContext(&clusterContextInfo, &clusterContext);

  nvclusterlod::ContextCreateInfo lodContextInfo;
  nvclusterlod::Context           lodContext;

  lodContextInfo.clusterContext = clusterContext;
  nvclusterlodCreateContext(&lodContextInfo, &lodContext);

  uint64_t numTotalTriangles = 0;
  uint64_t numTotalStrips    = 0;

  bool     preferInnerParallelism = m_geometries.size() < nvh::get_thread_pool().get_thread_count() * 2;
  uint32_t numOuterThreads        = preferInnerParallelism ? 1 : nvh::get_thread_pool().get_thread_count();

  nvh::parallel_batches_indexed<1>(
      m_geometries.size(),
      [&](uint64_t idx, uint32_t threadIdx) {
        Geometry&                     geom = m_geometries[idx];
        nvclusterlod::LodGeometryView view = {};

        if(cacheView.isValid())
        {
          cacheView.getLodGeometryView(view, idx);
        }

        buildGeometryClusters(lodContext, geom, view, preferInnerParallelism);

        // no longer need original triangles
        geom.globalTriangles = {};

        if(geom.lodMesh.clusterTriangleRanges.empty())
          return;

        if(m_config.clusterStripify)
        {
          buildGeometryClusterStrips(geom, numTotalTriangles, numTotalStrips, preferInnerParallelism);
        }

        buildGeometryClusterVertices(geom, preferInnerParallelism);

        // no longer need vertex indirection
        geom.localVertices = {};

        buildGeometryBboxes(geom, preferInnerParallelism);

        m_clusterMaxVerticesCount = std::max(m_clusterMaxVerticesCount, geom.clusterMaxVerticesCount);
      },
      numOuterThreads);

  // reset settings in case we had a valid cache file
  if(cacheView.isValid())
  {
    // override settings
    m_config.clusterTriangles         = 0;
    m_config.clusterGroupSize         = 0;
    m_config.lodLevelDecimationFactor = 0;

    for(size_t g = 0; g < m_geometries.size(); g++)
    {
      Geometry& geom = m_geometries[g];

      m_config.clusterTriangles = std::max(m_config.clusterTriangles, geom.lodInfo.clusterConfig.maxClusterSize);
      m_config.clusterGroupSize = std::max(m_config.clusterGroupSize, geom.lodInfo.groupConfig.maxClusterSize);
      m_config.lodLevelDecimationFactor = std::max(m_config.lodLevelDecimationFactor, geom.lodInfo.decimationFactor);
    }
  }

  nvclusterlodDestroyContext(lodContext);
  nvclusterDestroyContext(clusterContext);

  double endTime = clock.getMicroSeconds();
  LOGI("Scene cluster build time: %f milliseconds\n", (endTime - startTime) / 1000.0f);


  if(m_config.clusterStripify && (numTotalStrips > 0))
  {
    LOGI("Average triangles per strip %.2f\n", double(numTotalTriangles) / double(numTotalStrips));
  }

  computeHistograms();
}

void Scene::buildGeometryClusters(nvclusterlod::Context lodcontext, Geometry& geom, const nvclusterlod::LodGeometryView& view, bool doParallel)
{
  nvclusterlod::Result result;

  nvclusterlod::MeshInput lodMeshInput;
  lodMeshInput.decimationFactor = m_config.lodLevelDecimationFactor;
  lodMeshInput.indexCount       = uint32_t(geom.globalTriangles.size() * 3);
  lodMeshInput.indices          = reinterpret_cast<const uint32_t*>(geom.globalTriangles.data());
  lodMeshInput.vertexCount      = uint32_t(geom.positions.size());
  lodMeshInput.vertexOffset     = 0;
  lodMeshInput.vertexStride     = sizeof(glm::vec3);
  lodMeshInput.vertices         = reinterpret_cast<const float*>(geom.positions.data());

  lodMeshInput.clusterConfig                = {};
  lodMeshInput.clusterConfig.minClusterSize = (m_config.clusterTriangles * 3) / 4;
  lodMeshInput.clusterConfig.maxClusterSize = m_config.clusterTriangles;
  lodMeshInput.groupConfig                  = {};
  lodMeshInput.groupConfig.minClusterSize   = (m_config.clusterGroupSize * 3) / 4;
  lodMeshInput.groupConfig.maxClusterSize   = m_config.clusterGroupSize;


  geom.lodInfo.clusterConfig            = lodMeshInput.clusterConfig;
  geom.lodInfo.groupConfig              = lodMeshInput.groupConfig;
  geom.lodInfo.decimationFactor         = lodMeshInput.decimationFactor;
  geom.lodInfo.inputTriangleCount       = geom.globalTriangles.size();
  geom.lodInfo.inputVertexCount         = geom.positions.size();
  geom.lodInfo.inputTriangleIndicesHash = 0;
  geom.lodInfo.inputVerticesHash        = 0;

  nvclusterlod::LodGeometryInfo compareInfo = geom.lodInfo;

  // if view has content
  if(view.info.inputTriangleCount)
  {
    // then always use what was stored in view for config
    compareInfo.clusterConfig    = view.info.clusterConfig;
    compareInfo.groupConfig      = view.info.groupConfig;
    compareInfo.decimationFactor = view.info.decimationFactor;
  }

  if(memcmp(&compareInfo, &view.info, sizeof(compareInfo)) == 0)
  {
    geom.lodInfo = view.info;
    nvclusterlod::toStorage(view.lodMesh, geom.lodMesh);
    nvclusterlod::toStorage(view.lodHierarchy, geom.lodHierachy);
  }
  else
  {
    m_prebuildLodClusters = false;

    result = nvclusterlod::generateLodMesh(lodcontext, lodMeshInput, geom.lodMesh);
    if(result != nvclusterlod::SUCCESS)
    {
      assert(0);
      LOGE("nvclusterlod::generateLodMesh failed: %d\n", result);
      std::exit(-1);
    }

    nvclusterlod::HierarchyInput hierarchyInput = {};
    hierarchyInput.clusterCount                 = uint32_t(geom.lodMesh.clusterBoundingSpheres.size());
    hierarchyInput.clusterBoundingSpheres       = geom.lodMesh.clusterBoundingSpheres.data();
    hierarchyInput.clusterGeneratingGroups      = geom.lodMesh.clusterGeneratingGroups.data();
    hierarchyInput.groupClusterRanges           = geom.lodMesh.groupClusterRanges.data();
    hierarchyInput.groupQuadricErrors           = geom.lodMesh.groupQuadricErrors.data();
    hierarchyInput.lodLevelGroupRanges          = geom.lodMesh.lodLevelGroupRanges.data();
    hierarchyInput.lodLevelCount                = uint32_t(geom.lodMesh.lodLevelGroupRanges.size());
    hierarchyInput.groupCount                   = int32_t(geom.lodMesh.groupClusterRanges.size());

    // required to later traverse the lod hierarchy in parallel
    // Note we build hierarchies over each lod level's groups
    // and then join them together to a single hierarchy.
    // This is key to the parallel traversal algorithm that traverse multiple lod levels
    // at once.

    result = nvclusterlod::generateLodHierarchy(lodcontext, hierarchyInput, geom.lodHierachy);
    if(result != nvclusterlod::SUCCESS)
    {
      assert(0);
      LOGE("nvclusterlod::generateLodHierarchy failed: %d\n", result);
      std::exit(-1);
    }
  }

  // required for streaming, need to ensure that a high-detail group
  result = nvclusterlod::generateGroupGeneratingGroups(geom.lodMesh.groupClusterRanges,
                                                       geom.lodMesh.clusterGeneratingGroups, geom.groupGeneratingGroups);
  if(result != nvclusterlod::SUCCESS)
  {
    assert(0);
    LOGE("nvclusterlod::generateGroupGeneratingGroups failed: %d\n", result);
    std::exit(-1);
  }

  // build localized index buffers

  geom.localTriangles.resize(geom.lodMesh.triangleVertices.size());
  geom.localVertices.resize(geom.lodMesh.triangleVertices.size());
  geom.clusterVertexRanges.resize(geom.lodMesh.clusterTriangleRanges.size());

  uint32_t              numThreads = doParallel ? nvh::get_thread_pool().get_thread_count() : 1;
  std::vector<uint32_t> threadClusterMaxVertices(numThreads, 0);
  std::vector<uint32_t> threadClusterMaxTriangles(numThreads, 0);

  nvh::parallel_batches_indexed(
      geom.lodMesh.clusterTriangleRanges.size(),
      [&](uint64_t idx, uint32_t threadIdx) {
        nvcluster::Range& vertexRange            = geom.clusterVertexRanges[idx];
        nvcluster::Range  triangleRange          = geom.lodMesh.clusterTriangleRanges[idx];
        const uint32_t* __restrict inputTriangle = &geom.lodMesh.triangleVertices[triangleRange.offset * 3];
        uint8_t* __restrict outputTriangle       = &geom.localTriangles[triangleRange.offset * 3];
        uint32_t* __restrict outVertices         = &geom.localVertices[triangleRange.offset * 3];
        uint32_t count                           = 0;

        for(uint32_t i = 0; i < triangleRange.count * 3; i++)
        {
          uint32_t vertexIndex = inputTriangle[i];
          uint32_t cacheIndex  = ~0;
          for(uint32_t v = 0; v < count; v++)
          {
            if(outVertices[v] == vertexIndex)
            {
              cacheIndex = v;
            }
          }
          if(cacheIndex == ~0)
          {
            cacheIndex              = count++;
            outVertices[cacheIndex] = vertexIndex;
          }
          outputTriangle[i] = uint8_t(cacheIndex);
        }

        vertexRange.count                    = count;
        threadClusterMaxVertices[threadIdx]  = std::max(threadClusterMaxVertices[threadIdx], vertexRange.count);
        threadClusterMaxTriangles[threadIdx] = std::max(threadClusterMaxTriangles[threadIdx], triangleRange.count);
      },
      numThreads);

  geom.clusterMaxVerticesCount  = 0;
  geom.clusterMaxTrianglesCount = 0;
  for(uint32_t t = 0; t < numThreads; t++)
  {
    geom.clusterMaxVerticesCount  = std::max(geom.clusterMaxVerticesCount, threadClusterMaxVertices[t]);
    geom.clusterMaxTrianglesCount = std::max(geom.clusterMaxTrianglesCount, threadClusterMaxTriangles[t]);
  }

  // compaction pass
  uint32_t* localVertices = geom.localVertices.data();

  uint32_t offset = 0;
  for(size_t c = 0; c < geom.lodMesh.clusterTriangleRanges.size(); c++)
  {
    nvcluster::Range& vertexRange   = geom.clusterVertexRanges[c];
    nvcluster::Range  triangleRange = geom.lodMesh.clusterTriangleRanges[c];

    vertexRange.offset = offset;

    memmove(&localVertices[offset], &localVertices[triangleRange.offset * 3], sizeof(uint32_t) * vertexRange.count);

    offset += vertexRange.count;
  }

  geom.localVertices.resize(offset);

  geom.lodLevelsCount = uint32_t(geom.lodMesh.lodLevelGroupRanges.size());

  // for later, easier access
  geom.groupLodLevels.resize(geom.lodMesh.groupClusterRanges.size());

  geom.hiClustersCount = 0;
  geom.hiTriangleCount = 0;
  geom.hiVerticesCount = 0;

  for(size_t level = 0; level < geom.lodMesh.lodLevelGroupRanges.size(); level++)
  {
    nvcluster::Range groupRange = geom.lodMesh.lodLevelGroupRanges[level];
    for(size_t g = groupRange.offset; g < groupRange.offset + groupRange.count; g++)
    {
      geom.groupLodLevels[g] = uint8_t(level);

      if(level == 0)
      {
        nvcluster::Range clusterRange = geom.lodMesh.groupClusterRanges[g];

        uint32_t lastCluster  = clusterRange.offset + clusterRange.count - 1;
        uint32_t firstCluster = clusterRange.offset;

        geom.hiClustersCount += clusterRange.count;
        geom.hiTriangleCount += geom.lodMesh.clusterTriangleRanges[lastCluster].count
                                + geom.lodMesh.clusterTriangleRanges[lastCluster].offset
                                - geom.lodMesh.clusterTriangleRanges[firstCluster].offset;
        geom.hiVerticesCount += geom.clusterVertexRanges[lastCluster].count + geom.clusterVertexRanges[lastCluster].offset
                                - geom.clusterVertexRanges[firstCluster].offset;
      }
    }
  }

  geom.totalTriangleCount = uint32_t(geom.localTriangles.size() / 3);
  geom.totalVerticesCount = uint32_t(geom.localVertices.size());
  geom.totalClustersCount = uint32_t(geom.lodMesh.clusterTriangleRanges.size());
}

void Scene::computeLodBboxes_recursive(Geometry& geom, size_t i)
{
  const nvclusterlod::Node& node = geom.lodHierachy.nodes[i];
  shaderio::BBox&           bbox = geom.nodeBboxes[i];

  bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f};

  if(node.children.isLeafNode)
  {
    nvcluster::Range clusterRange = geom.lodMesh.groupClusterRanges[node.clusters.group];
    for(size_t c = clusterRange.offset; c < clusterRange.offset + clusterRange.count; c++)
    {
      bbox.lo = glm::min(bbox.lo, geom.clusterBboxes[c].lo);
      bbox.hi = glm::max(bbox.hi, geom.clusterBboxes[c].hi);
    }
  }
  else
  {
    for(uint32_t n = 0; n <= node.children.childCountMinusOne; n++)
    {
      computeLodBboxes_recursive(geom, node.children.childOffset + n);
    }

    for(uint32_t n = 0; n <= node.children.childCountMinusOne; n++)
    {
      bbox.lo = glm::min(bbox.lo, geom.nodeBboxes[node.children.childOffset + n].lo);
      bbox.hi = glm::max(bbox.hi, geom.nodeBboxes[node.children.childOffset + n].hi);
    }
  }
}

void Scene::buildGeometryBboxes(Geometry& geom, bool doParallel)
{
  uint32_t numThreads = doParallel ? nvh::get_thread_pool().get_thread_count() : 1;

  geom.clusterBboxes.resize(geom.lodMesh.clusterTriangleRanges.size());

  const glm::vec3* positions      = geom.positions.data();
  const uint8_t*   localTriangles = geom.localTriangles.data();

  nvh::parallel_batches_indexed(
      geom.lodMesh.clusterTriangleRanges.size(),
      [&](uint64_t idx, uint32_t threadIdx) {
        nvcluster::Range& vertexRange   = geom.clusterVertexRanges[idx];
        nvcluster::Range& triangleRange = geom.lodMesh.clusterTriangleRanges[idx];

        shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, FLT_MAX, -FLT_MAX};
        for(uint32_t v = 0; v < vertexRange.count; v++)
        {
          uint32_t  vertexIndex = vertexRange.offset + v;
          glm::vec3 pos         = positions[vertexIndex];

          bbox.lo = glm::min(bbox.lo, pos);
          bbox.hi = glm::max(bbox.hi, pos);
        }

        // find longest edge
        for(uint32_t t = 0; t < triangleRange.count; t++)
        {
          glm::vec3 trianglePositions[3];

          for(uint32_t v = 0; v < 3; v++)
          {
            trianglePositions[v] = positions[uint32_t(localTriangles[(triangleRange.offset + t) * 3 + v]) + vertexRange.offset];
          }

          for(uint32_t e = 0; e < 3; e++)
          {
            float distance    = glm::distance(trianglePositions[e], trianglePositions[(e + 1) % 3]);
            bbox.shortestEdge = std::min(bbox.shortestEdge, distance);
            bbox.longestEdge  = std::max(bbox.longestEdge, distance);
          }
        }

        geom.clusterBboxes[idx] = bbox;
      },
      numThreads);


  // now build lod node bounding boxes
  geom.nodeBboxes.resize(geom.lodHierachy.nodes.size());
  computeLodBboxes_recursive(geom, 0);
}


void Scene::computeHistograms()
{
  m_clusterTriangleHistogram.resize(m_config.clusterTriangles + 1, 0);
  m_clusterVertexHistogram.resize(m_clusterMaxVerticesCount + 1, 0);
  m_groupClusterHistogram.resize(m_config.clusterGroupSize + 1, 0);
  m_nodeChildrenHistogram.resize(32 + 1, 0);

  for(Geometry& geom : m_geometries)
  {
    for(size_t c = 0; c < geom.lodMesh.clusterTriangleRanges.size(); c++)
    {
      const nvcluster::Range& vertexRange   = geom.clusterVertexRanges[c];
      const nvcluster::Range& triangleRange = geom.lodMesh.clusterTriangleRanges[c];

      m_clusterTriangleHistogram[triangleRange.count]++;
      m_clusterVertexHistogram[vertexRange.count]++;
    }

    for(size_t g = 0; g < geom.lodMesh.groupClusterRanges.size(); g++)
    {
      const nvcluster::Range& groupRange = geom.lodMesh.groupClusterRanges[g];
      if(groupRange.count + 1 > m_groupClusterHistogram.size())
      {
        m_groupClusterHistogram.resize(groupRange.count + 1, 0);
      }
      m_groupClusterHistogram[groupRange.count]++;
    }

    for(size_t n = 0; n < geom.lodHierachy.nodes.size(); n++)
    {
      const nvclusterlod::Node& node = geom.lodHierachy.nodes[n];
      if(node.children.isLeafNode)
      {
        continue;
      }

      if(node.children.childCountMinusOne + 1 + 1 > m_nodeChildrenHistogram.size())
      {
        m_nodeChildrenHistogram.resize(node.children.childCountMinusOne + 1 + 1, 0);
      }
      m_nodeChildrenHistogram[node.children.childCountMinusOne + 1]++;
    }
  }

  m_clusterTriangleHistogramMax = 0u;
  m_clusterVertexHistogramMax   = 0u;
  m_groupClusterHistogramMax    = 0u;
  m_nodeChildrenHistogramMax    = 0u;
  for(size_t i = 0; i < m_clusterTriangleHistogram.size(); i++)
  {
    m_clusterTriangleHistogramMax = std::max(m_clusterTriangleHistogramMax, m_clusterTriangleHistogram[i]);
  }
  for(size_t i = 0; i < m_clusterVertexHistogram.size(); i++)
  {
    m_clusterVertexHistogramMax = std::max(m_clusterVertexHistogramMax, m_clusterVertexHistogram[i]);
  }
  for(size_t i = 0; i < m_groupClusterHistogram.size(); i++)
  {
    m_groupClusterHistogramMax = std::max(m_groupClusterHistogramMax, m_groupClusterHistogram[i]);
  }
  for(size_t i = 0; i < m_nodeChildrenHistogram.size(); i++)
  {
    m_nodeChildrenHistogramMax = std::max(m_nodeChildrenHistogramMax, m_nodeChildrenHistogram[i]);
  }
}

void Scene::buildGeometryClusterStrips(Geometry& geom, uint64_t& totalTriangles, uint64_t& totalStrips, bool doParallel)
{
  uint32_t numThreads = doParallel ? nvh::get_thread_pool().get_thread_count() : 1;

  std::vector<std::vector<uint32_t>> threadIndices(numThreads);

  uint32_t numMaxTriangles = m_config.clusterTriangles;
  for(uint32_t t = 0; t < numThreads; t++)
  {
    threadIndices[t].resize(numMaxTriangles * 3 + numMaxTriangles * 3 + meshopt_stripifyBound(numMaxTriangles * 3));
  }

  std::atomic_uint32_t numStrips = 0;

  uint8_t* localTriangles = geom.localTriangles.data();

  nvh::parallel_batches_indexed(
      geom.lodMesh.clusterTriangleRanges.size(),
      [&](uint64_t idx, uint32_t threadIdx) {
        nvcluster::Range triangleRange = geom.lodMesh.clusterTriangleRanges[idx];
        nvcluster::Range vertexRange   = geom.clusterVertexRanges[idx];

        uint32_t* meshletIndices      = threadIndices[threadIdx].data();
        uint32_t* meshletOptim        = meshletIndices + triangleRange.count * 3;
        uint32_t* meshletStripIndices = meshletOptim + triangleRange.count * 3;

        // convert u8 to u32
        for(uint32_t i = 0; i < triangleRange.count * 3; i++)
        {
          meshletIndices[i] = localTriangles[triangleRange.offset * 3 + i];
        }

        meshopt_optimizeVertexCache(meshletOptim, meshletIndices, triangleRange.count * 3, vertexRange.count);
        size_t stripIndexCount =
            meshopt_stripify(meshletStripIndices, meshletOptim, triangleRange.count * 3, vertexRange.count, ~0);
        size_t newIndexCount = meshopt_unstripify(meshletIndices, meshletStripIndices, stripIndexCount, ~0);

        triangleRange.count = uint32_t(newIndexCount / 3);

        for(uint32_t i = 0; i < uint32_t(newIndexCount); i++)
        {
          localTriangles[triangleRange.offset * 3 + i] = uint8_t(meshletIndices[i]);
        }

        // just for stats
        numStrips++;
        for(uint32_t t = 1; t < uint32_t(triangleRange.count); t++)
        {
          const uint32_t* current = meshletIndices + t * 3;
          const uint32_t* prev    = meshletIndices + (t - 1) * 3;

          if(!((current[0] == prev[0] || current[0] == prev[2]) && (current[1] == prev[1] || current[1] == prev[2])))
            numStrips++;
        }
      },
      numThreads);

  totalTriangles += geom.localTriangles.size();
  totalStrips += numStrips;
}

void Scene::buildGeometryClusterVertices(Geometry& geom, bool doParallel)
{
  uint32_t numThreads = doParallel ? nvh::get_thread_pool().get_thread_count() : 1;

  // build per-cluster vertices

  std::vector<glm::vec3> oldPositionsData = std::move(geom.positions);
  std::vector<glm::vec3> oldNormalsData   = std::move(geom.normals);

  uint32_t*      localVertices  = geom.localVertices.data();
  const uint8_t* localTriangles = geom.localTriangles.data();

  geom.positions.resize(geom.localVertices.size());
  geom.normals.resize(geom.localVertices.size());

  const glm::vec3* oldPositions = oldPositionsData.data();
  const glm::vec3* oldNormals   = oldNormalsData.data();
  glm::vec3*       newPositions = geom.positions.data();
  glm::vec3*       newNormals   = geom.normals.data();

  nvh::parallel_batches_indexed(
      geom.clusterVertexRanges.size(),
      [&](uint64_t c, uint32_t threadIdx) {
        nvcluster::Range& vertexRange = geom.clusterVertexRanges[c];

        for(uint32_t v = 0; v < vertexRange.count; v++)
        {
          uint32_t oldIdx                       = localVertices[v + vertexRange.offset];
          localVertices[v + vertexRange.offset] = v + vertexRange.offset;
          newPositions[v + vertexRange.offset]  = oldPositions[oldIdx];
          newNormals[v + vertexRange.offset]    = oldNormals[oldIdx];
        }
      },
      numThreads);
}


}  // namespace lodclusters
