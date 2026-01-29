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

#include <cinttypes>
#include <cstring>
#include <random>

#include <meshoptimizer.h>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/hash_operations.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "scene.hpp"


namespace lodclusters {

void Scene::ProcessingInfo::init(float processingThreadsPct)
{
  numPoolThreadsOriginal = nvutils::get_thread_pool().get_thread_count();

  numPoolThreads = numPoolThreadsOriginal;
  if(processingThreadsPct > 0.0f && processingThreadsPct < 1.0f)
  {
    numPoolThreads = std::min(numPoolThreads, std::max(1u, uint32_t(ceilf(float(numPoolThreads) * processingThreadsPct))));

    if(numPoolThreads != numPoolThreadsOriginal)
      nvutils::get_thread_pool().reset(numPoolThreads);
  }
}

void Scene::ProcessingInfo::setupParallelism(size_t geometryCount_, size_t geometryCompletedCount, int parallelismMode)
{
  geometryCount = geometryCount_;

  bool preferInnerParallelism = (geometryCount - geometryCompletedCount) < numPoolThreads;

  if(parallelismMode < 0)
  {
    preferInnerParallelism = true;
  }
  if(parallelismMode > 0)
  {
    preferInnerParallelism = false;
  }

  numOuterThreads = preferInnerParallelism ? 1 : numPoolThreads;
  numInnerThreads = preferInnerParallelism ? numPoolThreads : 1;
}

void Scene::ProcessingInfo::setupCompressedGltf(size_t bufferViewCount)
{
  bufferViewUsers.resize(bufferViewCount, {0});
  bufferViewLocks.resize(bufferViewCount, {0});
}

void Scene::ProcessingInfo::logBegin(uint64_t totalTriangleCount)
{
  LOGI("... geometry load & processing: geometries %" PRIu64 ", threads outer %d inner %d\n", geometryCount,
       numOuterThreads, numInnerThreads);

  startTime = clock.getMicroseconds();

  triangleCount               = totalTriangleCount;
  progressTrianglesCompleted  = 0;
  progressGeometriesCompleted = 0;
  progressLastPercentage      = 0;
}

uint32_t Scene::ProcessingInfo::logCompletedGeometry(uint64_t geometryTriangleCount)
{
  std::lock_guard lock(progressMutex);

  progressGeometriesCompleted++;
  progressTrianglesCompleted += geometryTriangleCount;

  uint32_t percentage;
  if(!triangleCount)
  {
    percentage = uint32_t(double(progressGeometriesCompleted * 100) / double(geometryCount));
  }
  else
  {
    percentage = uint32_t((double(progressTrianglesCompleted) * 100) / double(triangleCount));
  }

  // statistics
  const uint32_t precentageGranularity = 5;
  uint32_t       percentageSnapped     = (percentage / precentageGranularity) * precentageGranularity;

  if(percentageSnapped > progressLastPercentage)
  {
    progressLastPercentage = percentageSnapped;
    LOGI("... geometry load & processing: %3d%%\n", percentageSnapped);
  }

  return percentage;
}

void Scene::ProcessingInfo::logEnd()
{
  double endTime = clock.getMicroseconds();

  LOGI("... geometry load & processing: %f milliseconds\n", (endTime - startTime) / 1000.0f);

  // can be zero if loaded from cache
  if(stats.groups)
  {
    LOGI("Group Data Stats\n");
    LOGI("Groups:               %12" PRIu64 "\n", (uint64_t)stats.groups);
    LOGI("Clusters:             %12" PRIu64 "\n", (uint64_t)stats.clusters);
    LOGI("Vertices:             %12" PRIu64 "\n", (uint64_t)stats.vertices);
    LOGI("Group Unique Verts:   %12" PRIu64 "\n", (uint64_t)stats.groupUniqueVertices);
    LOGI("Group Header Bytes:   %12" PRIu64 "\n", (uint64_t)stats.groupHeaderBytes);
    LOGI("Cluster Header Bytes: %12" PRIu64 "\n", (uint64_t)stats.clusterHeaderBytes);
    LOGI("Cluster BBox Bytes:   %12" PRIu64 "\n", (uint64_t)stats.clusterBboxBytes);
    LOGI("Cluster GGrp Bytes:   %12" PRIu64 "\n", (uint64_t)stats.clusterGenBytes);
    LOGI("Triangle Index Bytes: %12" PRIu64 "\n", (uint64_t)stats.triangleIndexBytes);
    LOGI("Vertex All Bytes:     %12" PRIu64 "\n", (uint64_t)(stats.vertexPosBytes + stats.vertexNrmBytes + stats.vertexTexCoordBytes));
    LOGI("Vertex Pos Bytes:     %12" PRIu64 "\n", (uint64_t)stats.vertexPosBytes);
    LOGI("Vertex TexCrd Bytes:  %12" PRIu64 "\n", (uint64_t)stats.vertexTexCoordBytes);
    LOGI("Vertex N&T Bytes:     %12" PRIu64 "\n", (uint64_t)stats.vertexNrmBytes);
    LOGI("Vertex Comp Bytes:    %12" PRIu64 "\n", (uint64_t)stats.vertexCompressedBytes);
    LOGI("\n");
  }
}

void Scene::ProcessingInfo::deinit()
{
  if(numPoolThreads != numPoolThreadsOriginal)
    nvutils::get_thread_pool().reset(numPoolThreadsOriginal);
}

void Scene::fillGroupRuntimeData(const GroupInfo& srcGroupInfo,
                                 const GroupView& srcGroupView,
                                 uint32_t         groupID,
                                 uint32_t         groupResidentID,
                                 uint32_t         clusterResidentID,
                                 void*            dst,
                                 size_t           dstSize)
{
  GroupInfo dstGroupInfo = srcGroupInfo;
  if(srcGroupInfo.uncompressedSizeBytes)
  {
    decompressGroup(srcGroupInfo, srcGroupView, dst, dstSize);

    dstGroupInfo.sizeBytes       = dstGroupInfo.uncompressedSizeBytes;
    dstGroupInfo.vertexDataCount = dstGroupInfo.uncompressedVertexDataCount;
  }
  else
  {
    assert(srcGroupView.rawSize <= dstSize);
    memcpy(dst, srcGroupView.raw, srcGroupView.rawSize);
  }

  // final patching
  {
    GroupStorage groupStorage(dst, dstGroupInfo);
    groupStorage.group->residentID        = groupResidentID;
    groupStorage.group->clusterResidentID = clusterResidentID;
  }
}

Scene::Result Scene::init(const std::filesystem::path& filePath,
                          const SceneConfig&           config,
                          const SceneLoaderConfig&     loaderConfig,
                          const std::string&           cacheSuffix,
                          bool                         skipCache)
{
  *this = {};

  m_filePath             = filePath;
  m_config               = config;
  m_loaderConfig         = loaderConfig;
  m_loadedFromCache      = false;
  m_cacheFilePath        = filePath;
  m_cachePartialFilePath = filePath;
  m_cacheFileSize        = 0;
  m_cacheSuffix          = cacheSuffix;

  std::string oldExtension = filePath.extension().string();
  m_cacheFilePath.replace_extension(oldExtension + cacheSuffix);
  m_cachePartialFilePath.replace_extension(oldExtension + cacheSuffix + "_partial");

  if(!skipCache && !m_loaderConfig.processingOnly && m_loaderConfig.autoLoadCache)
  {
    openCache();
  }

  ProcessingInfo processingInfo;
  processingInfo.init(m_loaderConfig.processingThreadsPct);

  Result loadResult = loadGLTF(processingInfo, filePath);
  if(loadResult == SCENE_RESULT_NEEDS_PREPROCESS || loadResult == SCENE_RESULT_CACHE_INVALID)
  {
    LOGI("Scene::init large scene or invalid cache detected\n  using dedicated preprocess pass\n");
    closeCache();

    m_loaderConfig.processingOnly = true;
    loadResult                    = loadGLTF(processingInfo, filePath);
    m_loaderConfig.processingOnly = false;
    if(loadResult == SCENE_RESULT_PREPROCESS_COMPLETED)
    {
      openCache();
      loadResult = loadGLTF(processingInfo, filePath);
    }
  }

  processingInfo.deinit();

  if(loadResult != SCENE_RESULT_SUCCESS)
  {
    closeCache();

    return loadResult;
  }

  if(m_loadedFromCache)
  {
    m_cacheFileView.getSceneConfig(m_config);
    m_cacheFileView.getHistograms(m_histograms);
  }

  m_originalInstanceCount = m_instances.size();
  m_originalGeometryCount = m_geometryViews.size();
  m_activeGeometryCount   = m_originalGeometryCount;

  computeInstanceBBoxes();
  m_gridBbox = m_bbox;

  glm::vec3 modelExtent = m_bbox.hi - m_bbox.lo;
  m_isBig = modelExtent.y < 0.15f * std::max(modelExtent.x, modelExtent.z) && m_originalInstanceCount > 1024;

  for(auto& geometry : m_geometryViews)
  {
    m_hiPerGeometryTriangles  = std::max(m_hiPerGeometryTriangles, geometry.hiTriangleCount);
    m_hiPerGeometryVertices   = std::max(m_hiPerGeometryVertices, geometry.hiVerticesCount);
    m_hiPerGeometryClusters   = std::max(m_hiPerGeometryClusters, geometry.hiClustersCount);
    m_maxPerGeometryTriangles = std::max(m_maxPerGeometryTriangles, geometry.totalTriangleCount);
    m_maxPerGeometryVertices  = std::max(m_maxPerGeometryVertices, geometry.totalVerticesCount);
    m_maxPerGeometryClusters  = std::max(m_maxPerGeometryClusters, geometry.totalClustersCount);
    m_maxClusterVertices      = std::max(m_maxClusterVertices, geometry.clusterMaxVerticesCount);
    m_maxClusterTriangles     = std::max(m_maxClusterTriangles, geometry.clusterMaxTrianglesCount);
    m_maxLodLevelsCount       = std::max(m_maxLodLevelsCount, geometry.lodLevelsCount);

    m_hiTrianglesCount += geometry.hiTriangleCount;
    m_hiClustersCount += geometry.hiClustersCount;
    m_totalClustersCount += geometry.totalClustersCount;
    m_totalTrianglesCount += geometry.totalTriangleCount;
    m_totalVerticesCount += geometry.totalVerticesCount;
  }
  for(size_t i = 0; i < m_instances.size(); i++)
  {
    const GeometryView& geometry = m_geometryViews[m_instances[i].geometryID];
    m_hiTrianglesCountInstanced += geometry.hiTriangleCount;
    m_hiClustersCountInstanced += geometry.hiClustersCount;
  }

  LOGI("clusters:  %" PRIu64 "\n", m_totalClustersCount);
  LOGI("triangles: %" PRIu64 "\n", m_totalTrianglesCount);
  LOGI("triangles/cluster: %.2f\n", double(m_totalTrianglesCount) / double(m_totalClustersCount));
  LOGI("vertices: %" PRIu64 "\n", m_totalVerticesCount);
  LOGI("vertices/cluster: %.2f\n", double(m_totalVerticesCount) / double(m_totalClustersCount));
  LOGI("hi clusters:  %" PRIu64 "\n", m_hiClustersCount);
  LOGI("hi triangles: %" PRIu64 "\n", m_hiTrianglesCount);
  LOGI("hi triangles/cluster: %.2f\n", double(m_hiTrianglesCount) / double(m_hiClustersCount));

  if(!m_loadedFromCache && m_loaderConfig.autoSaveCache)
  {
    saveCache();
  }

  if(m_loadedFromCache && !m_loaderConfig.memoryMappedCache)
  {
    // everything was loaded into system memory,
    // close file mappings
    closeCache();
  }

  return loadResult;
}

void Scene::deinit()
{
  *this = {};
}

void Scene::updateSceneGrid(const SceneGridConfig& gridConfig)
{
  m_gridConfig = gridConfig;

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

  glm::vec3 modelExtent = (m_bbox.hi - m_bbox.lo);
  glm::vec3 modelCenter = (m_bbox.hi + m_bbox.lo) * 0.5f;
  float     modelSize   = glm::length(modelExtent);
  glm::vec3 gridShift;
  glm::mat4 gridRotMatrix;

  for(size_t copyIndex = 1; copyIndex < copiesCount; copyIndex++)
  {
    gridShift = gridConfig.refShift * modelExtent;
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

    glm::mat4 scaleMatrix = glm::mat4(1.0f);

    if(gridConfig.minScale != 1.0f || gridConfig.maxScale != 1.0f)
    {
      float scale = glm::mix(gridConfig.minScale, gridConfig.maxScale, randomUnorm(rng));
      scaleMatrix = glm::scale(scaleMatrix, glm::vec3(scale));

      if(scale < 1.0f)
      {
        gridShift.y += modelSize * (1.0f - scale);
      }
      else
      {
        gridShift.y -= modelSize * (scale - 1.0f);
      }
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
      glm::vec3 translation = worldMatrix[3];
      worldMatrix[3]        = glm::vec4(translation - modelCenter, 1.f);

      worldMatrix = scaleMatrix * worldMatrix;

      if(axis & (8 | 16 | 32))
      {
        worldMatrix = gridRotMatrix * worldMatrix;
      }
      translation    = worldMatrix[3];
      worldMatrix[3] = glm::vec4(translation + modelCenter + gridShift, 1.f);

      instance.matrix = worldMatrix;
    }
  }

  m_gridBbox = m_bbox;
  computeInstanceBBoxes();
  std::swap(m_gridBbox, m_bbox);
}

void Scene::computeInstanceBBoxes()
{
  m_bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  for(auto& instance : m_instances)
  {
    const GeometryView& geometry = getActiveGeometry(instance.geometryID);

    instance.bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

    for(uint32_t v = 0; v < 8; v++)
    {
      bool x = (v & 1) != 0;
      bool y = (v & 2) != 0;
      bool z = (v & 4) != 0;

      glm::bvec3 weight(x, y, z);
      glm::vec3  corner = glm::mix(geometry.bbox.lo, geometry.bbox.hi, weight);
      corner            = instance.matrix * glm::vec4(corner, 1.0f);
      instance.bbox.lo  = glm::min(instance.bbox.lo, corner);
      instance.bbox.hi  = glm::max(instance.bbox.hi, corner);
    }

    m_bbox.lo = glm::min(m_bbox.lo, instance.bbox.lo);
    m_bbox.hi = glm::max(m_bbox.hi, instance.bbox.hi);
  }
}

void Scene::processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached)
{
  GeometryStorage& geometryStorage = m_geometryStorages[geometryIndex];
  GeometryView&    geometryView    = m_geometryViews[geometryIndex];

  bool viewFromStorage = true;

  if(isCached)
  {
    if(m_loaderConfig.memoryMappedCache)
    {
      m_cacheFileView.getGeometryView(geometryView, geometryIndex);

      viewFromStorage = false;
    }
    else
    {
      loadCachedGeometry(geometryStorage, geometryIndex);
    }
  }
  else
  {
    if(geometryStorage.triangles.empty())
    {
      geometryStorage = {};
    }
    else
    {
      // for cache file
      // The dedup might change the vertex count of the mesh, but for the cache file
      // comparison we actually want to use the original vertex count
      geometryStorage.lodInfo.inputTriangleCount       = geometryStorage.triangles.size();
      geometryStorage.lodInfo.inputVertexCount         = geometryStorage.vertexPositions.size();
      geometryStorage.lodInfo.inputTriangleIndicesHash = 0;
      geometryStorage.lodInfo.inputVerticesHash        = 0;

      size_t originalVertexCount = geometryStorage.vertexPositions.size();

      // some exports give us independent triangles, clean those up
      if(geometryStorage.vertexPositions.size() >= (geometryStorage.triangles.size() + geometryStorage.triangles.size() / 2))
      {
        buildGeometryDedupVertices(processingInfo, geometryStorage);
      }

      buildGeometryLod(processingInfo, geometryStorage);
    }
  }

  if(viewFromStorage)
  {
    (GeometryBase&)geometryView = geometryStorage;

    geometryView.groupData        = geometryStorage.groupData;
    geometryView.groupInfos       = geometryStorage.groupInfos;
    geometryView.lodLevels        = geometryStorage.lodLevels;
    geometryView.lodNodes         = geometryStorage.lodNodes;
    geometryView.lodNodeBboxes    = geometryStorage.lodNodeBboxes;
    geometryView.localMaterialIDs = geometryStorage.localMaterialIDs;
  }

  // always reset
  geometryView.instanceReferenceCount = 0;

  if(m_processingOnlyFile)
  {
    saveProcessingOnly(processingInfo, geometryIndex);
  }
}

void Scene::computeLodBboxes_recursive(GeometryStorage& geometry, size_t i)
{
  const shaderio::Node& node = geometry.lodNodes[i];
  shaderio::BBox&       bbox = geometry.lodNodeBboxes[i];

  bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f};

  if(node.groupRange.isGroup)
  {
    GroupInfo groupInfo = geometry.groupInfos[node.groupRange.groupIndex];
    GroupView groupView(geometry.groupData, groupInfo);

    for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
    {
      bbox.lo = glm::min(bbox.lo, groupView.clusterBboxes[c].lo);
      bbox.hi = glm::max(bbox.hi, groupView.clusterBboxes[c].hi);
    }
  }
  else
  {
    ((std::atomic_uint32_t&)m_histograms.nodeChildren[node.nodeRange.childCountMinusOne + 1])++;

    for(uint32_t n = 0; n <= node.nodeRange.childCountMinusOne; n++)
    {
      computeLodBboxes_recursive(geometry, node.nodeRange.childOffset + n);
    }

    for(uint32_t n = 0; n <= node.nodeRange.childCountMinusOne; n++)
    {
      bbox.lo = glm::min(bbox.lo, geometry.lodNodeBboxes[node.nodeRange.childOffset + n].lo);
      bbox.hi = glm::max(bbox.hi, geometry.lodNodeBboxes[node.nodeRange.childOffset + n].hi);
    }
  }
}

struct HashVertexRange
{
  uint32_t offset = 0;
  uint32_t count  = 0;
};

static_assert(std::atomic_uint32_t::is_always_lock_free && sizeof(std::atomic_uint32_t) == sizeof(uint32_t));

void Scene::buildGeometryDedupVertices(ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  std::vector<uint32_t> remap(geometry.vertexPositions.size());

  size_t uniqueVertices = 0;

  size_t attributeStride = geometry.vertexAttributes.size() / geometry.vertexPositions.size();

  if(geometry.attributeBits)
  {
    uint32_t       texOffset = 0;
    meshopt_Stream streams[4];
    uint32_t       streamCount = 1;

    streams[0].data   = geometry.vertexPositions.data();
    streams[0].size   = sizeof(float) * 3;
    streams[0].stride = sizeof(glm::vec3);
    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {
      streams[1].data   = geometry.vertexAttributes.data();
      streams[1].size   = sizeof(float) * 3;
      streams[1].stride = sizeof(float) * attributeStride;
      streamCount++;
      texOffset = 3;
    }
    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
    {
      streams[streamCount].data   = geometry.vertexAttributes.data() + texOffset;
      streams[streamCount].size   = sizeof(float) * 2;
      streams[streamCount].stride = sizeof(float) * attributeStride;
      streamCount++;
    }
    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
    {
      streams[streamCount].data   = geometry.vertexAttributes.data() + texOffset + 2;
      streams[streamCount].size   = sizeof(float) * 4;
      streams[streamCount].stride = sizeof(float) * attributeStride;
      streamCount++;
    }

    uniqueVertices =
        meshopt_generateVertexRemapMulti(remap.data(), reinterpret_cast<const uint32_t*>(geometry.triangles.data()),
                                         geometry.triangles.size() * 3, geometry.vertexPositions.size(), streams, streamCount);
  }
  else
  {
    uniqueVertices = meshopt_generateVertexRemap(remap.data(), reinterpret_cast<const uint32_t*>(geometry.triangles.data()),
                                                 geometry.triangles.size() * 3, geometry.vertexPositions.data(),
                                                 geometry.vertexPositions.size(), sizeof(glm::vec3));
  }

  {
    std::vector<glm::vec3> newPositions(uniqueVertices);
    meshopt_remapVertexBuffer(newPositions.data(), geometry.vertexPositions.data(), geometry.vertexPositions.size(),
                              sizeof(glm::vec3), remap.data());
    geometry.vertexPositions = std::move(newPositions);
  }

  if(geometry.attributeBits)
  {
    std::vector<float> newAttributes(uniqueVertices * attributeStride);
    meshopt_remapVertexBuffer(newAttributes.data(), geometry.vertexAttributes.data(), geometry.vertexPositions.size(),
                              sizeof(float) * attributeStride, remap.data());
    geometry.vertexAttributes = std::move(newAttributes);
  }

  meshopt_remapIndexBuffer(reinterpret_cast<uint32_t*>(geometry.triangles.data()),
                           reinterpret_cast<uint32_t*>(geometry.triangles.data()), geometry.triangles.size() * 3, remap.data());
}

void Scene::computeHistogramMaxs()
{
  m_histograms.clusterTrianglesMax = 0u;
  m_histograms.clusterVerticesMax  = 0u;
  m_histograms.groupClustersMax    = 0u;
  m_histograms.nodeChildrenMax     = 0u;
  m_histograms.lodLevelsMax        = 0u;

  for(size_t i = 0; i < m_histograms.clusterTriangles.size(); i++)
  {
    m_histograms.clusterTrianglesMax = std::max(m_histograms.clusterTrianglesMax, m_histograms.clusterTriangles[i]);
  }
  for(size_t i = 0; i < m_histograms.clusterVertices.size(); i++)
  {
    m_histograms.clusterVerticesMax = std::max(m_histograms.clusterVerticesMax, m_histograms.clusterVertices[i]);
  }

  for(size_t i = 0; i < m_histograms.groupClusters.size(); i++)
  {
    m_histograms.groupClustersMax = std::max(m_histograms.groupClustersMax, m_histograms.groupClusters[i]);
  }

  for(size_t i = 0; i < m_histograms.nodeChildren.size(); i++)
  {
    m_histograms.nodeChildrenMax = std::max(m_histograms.nodeChildrenMax, m_histograms.nodeChildren[i]);
  }

  for(size_t i = 0; i < m_histograms.lodLevels.size(); i++)
  {
    m_histograms.lodLevelsMax = std::max(m_histograms.lodLevelsMax, m_histograms.lodLevels[i]);
  }
}
}  // namespace lodclusters
