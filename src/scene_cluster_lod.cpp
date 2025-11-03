
/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <glm/gtc/constants.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <meshoptimizer.h>

#include "scene.hpp"
#include "../shaders/octant_encoding.h"
#include "../shaders/tangent_encoding.h"

namespace lodclusters {

void Scene::buildGeometryClusterLod(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  if(m_config.useNvLib)
  {
    buildGeometryClusterLodNvLib(processingInfo, geometry);
  }
  else
  {
    buildGeometryClusterLodMeshoptimizer(processingInfo, geometry);
  }

  geometry.lodNodeBboxes.resize(geometry.lodNodes.size());
  computeLodBboxes_recursive(geometry, 0);
}

// Takes the resulting cluster group of the lod generation and stores it into
// the internal representation used at runtime. This data is saved
// as is into the scene cache file and patched after upload when streamed in.
// Some abstraction is used to deal with results from either `meshoptimizer's` clusterlod,
// or `nv_cluster_lod_builder`.
uint32_t Scene::storeGroup(TempContext*                         context,
                           uint32_t                             threadIndex,
                           uint32_t                             groupIndex,
                           const TempGroup&                     group,
                           std::function<TempCluster(uint32_t)> tempClusterFn)
{
  GeometryStorage& geometry  = context->geometry;
  Scene::GroupInfo groupInfo = {};

  uint32_t level        = uint32_t(group.lodLevel);
  uint32_t clusterCount = uint32_t(group.clusterCount);

  uint8_t* groupTempData = &context->threadGroupDatas[context->threadGroupSize * threadIndex];

  Scene::GroupInfo groupTempInfo = context->threadGroupInfo;
  GroupStorage     groupTempStorage(groupTempData, groupTempInfo);

  std::span<uint32_t> vertexCacheEarlyValue((uint32_t*)(groupTempData + context->threadGroupStorageSize), 256);
  std::span<uint32_t> vertexCacheEarlyPos((uint32_t*)vertexCacheEarlyValue.data() + 256, 256);
  std::span<uint32_t> localVertices(vertexCacheEarlyPos.data() + 256, 256);

  uint32_t       clusterMaxVerticesCount  = 0;
  uint32_t       clusterMaxTrianglesCount = 0;
  shaderio::BBox groupBbox                = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  {
    // compute storage size, need to de-duplicate all vertices

    uint32_t triangleOffset   = 0;
    uint32_t vertexOffset     = 0;
    uint32_t vertexDataOffset = 0;

    uint32_t attributeStride = uint32_t(geometry.vertexAttributes.size() / geometry.vertexPositions.size());

    for(uint32_t c = 0; c < clusterCount; c++)
    {
      TempCluster tempCluster = tempClusterFn(c);

      shaderio::Cluster& groupCluster  = groupTempStorage.clusters[c];
      uint32_t           triangleCount = tempCluster.indexCount / 3;
      uint32_t           vertexCount   = 0;

      groupCluster.vertices = vertexDataOffset;
      groupCluster.indices  = triangleOffset * 3;

      memset(vertexCacheEarlyValue.data(), ~0, vertexCacheEarlyValue.size_bytes());
      for(uint32_t i = 0; i < tempCluster.indexCount; i++)
      {
        uint32_t vertexIndex = tempCluster.indices[i];
        uint32_t cacheIndex  = ~0;

        // quick early out, have we seen the index
        uint32_t cacheEarlyValue = vertexCacheEarlyValue[vertexIndex & 0xFF];
        if(cacheEarlyValue == vertexIndex)
        {
          cacheIndex = vertexCacheEarlyPos[vertexIndex & 0xFF];
        }
        else
        {
          // look for it serially
          for(uint32_t v = 0; v < vertexCount; v++)
          {
            if(localVertices[v] == vertexIndex)
            {
              cacheIndex = v;
            }
          }
        }

        if(cacheIndex == ~0)
        {
          cacheIndex                                = vertexCount++;
          localVertices[cacheIndex]                 = vertexIndex;
          vertexCacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;
          vertexCacheEarlyPos[vertexIndex & 0xFF]   = cacheIndex;
        }
        groupTempStorage.indices[i + triangleOffset * 3] = uint8_t(cacheIndex);
      }

      shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, FLT_MAX, -FLT_MAX};

      {
        for(uint32_t v = 0; v < vertexCount; v++)
        {
          // copy position
          glm::vec3 pos = geometry.vertexPositions[localVertices[v]];
          *(glm::vec3*)&groupTempStorage.vertices[vertexDataOffset + v * 3] = pos;

          // local bbox
          bbox.lo = glm::min(bbox.lo, glm::vec3(pos));
          bbox.hi = glm::max(bbox.hi, glm::vec3(pos));
        }
        vertexDataOffset += vertexCount * 3;
      }

      if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
      {
        for(uint32_t v = 0; v < vertexCount; v++)
        {
          glm::vec3 tmp =
              *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
          uint32_t encoded                                             = shaderio::vec_to_oct32(tmp);
          *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
        }

        vertexDataOffset += vertexCount;
      }

      if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_UV)
      {
        // align to vec2
        vertexDataOffset = (vertexDataOffset + 1) & ~1;

        for(uint32_t v = 0; v < vertexCount; v++)
        {
          glm::vec2 tmp =
              *(const glm::vec2*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeUvOffset]);
          *(glm::vec2*)&groupTempStorage.vertices[vertexDataOffset + v * 2] = tmp;
        }
        vertexDataOffset += vertexCount * 2;
      }

      if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
      {
        // last element might be partially written, clear to zero
        uint32_t elementCount                                          = (vertexCount + 1) / 2;
        groupTempStorage.vertices[vertexDataOffset + elementCount - 1] = 0;

        for(uint32_t v = 0; v < vertexCount && false; v++)
        {
          glm::vec4 userTangent =
              *(const glm::vec4*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeTangentOffset]);
          glm::vec3 userNormal = *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride]);

          uint16_t encoded = shaderio::tangent_pack(userNormal, userTangent);

          // put sign in lowest bit
          ((uint16_t*)&groupTempStorage.vertices[vertexDataOffset])[v] = encoded;
        }

        vertexDataOffset += elementCount;
      }

      // find longest edge
      for(uint32_t t = 0; t < triangleCount; t++)
      {
        glm::vec3 trianglePositions[3];

        for(uint32_t v = 0; v < 3; v++)
        {
          trianglePositions[v] =
              geometry.vertexPositions[localVertices[groupTempStorage.indices[(triangleOffset + t) * 3 + v]]];
        }

        for(uint32_t e = 0; e < 3; e++)
        {
          float distance    = glm::distance(trianglePositions[e], trianglePositions[(e + 1) % 3]);
          bbox.shortestEdge = std::min(bbox.shortestEdge, distance);
          bbox.longestEdge  = std::max(bbox.longestEdge, distance);
        }
      }

      groupBbox.lo = glm::min(groupBbox.lo, bbox.lo);
      groupBbox.hi = glm::max(groupBbox.hi, bbox.hi);

      groupTempStorage.clusterBboxes[c]           = bbox;
      groupTempStorage.clusterGeneratingGroups[c] = tempCluster.generatingGroup;

      groupCluster.triangleCountMinusOne = uint8_t(triangleCount - 1);
      groupCluster.vertexCountMinusOne   = uint8_t(vertexCount - 1);
      groupCluster.lodLevel              = uint8_t(level);
      groupCluster.groupChildIndex       = uint8_t(c);
      groupCluster.attributeBits         = uint8_t(geometry.attributeBits);
      groupCluster.localMaterialID       = uint8_t(0);
      groupCluster.reserved              = 0;

      clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, triangleCount);
      clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, vertexCount);

      vertexOffset += vertexCount;
      triangleOffset += triangleCount;
    }

    groupInfo.offsetBytes     = 0;
    groupInfo.clusterCount    = uint8_t(clusterCount);
    groupInfo.triangleCount   = uint16_t(triangleOffset);
    groupInfo.vertexCount     = uint16_t(vertexOffset);
    groupInfo.lodLevel        = uint8_t(level);
    groupInfo.attributeBits   = uint8_t(geometry.attributeBits);
    groupInfo.vertexDataCount = uint16_t(vertexDataOffset);
    groupInfo.sizeBytes       = groupInfo.computeSize();
  }

  // do actual storage & basic stats

  bool useOrderedLock = groupIndex != ~0 && context->innerThreadingActive;

  if(useOrderedLock)
  {
    // Want to enter the mutex in an ordered fashion
    // to preserve storage order from library.
    // It works without this as well, but then we don't have determinism in the
    // memory storage order of groups. And we might want to sort groups spatially
    // for more cache-efficient loading.
    while(true)
    {
      if(context->groupIndexOrdered.load() == groupIndex)
      {
        groupInfo.offsetBytes = context->groupDataOrdered.fetch_add(groupInfo.sizeBytes);

        context->groupIndexOrdered.store(groupIndex + 1);
        break;
      }
      else
      {
        std::this_thread::yield();
      }
    }
  }

  {
    std::lock_guard lock(context->groupMutex);

    geometry.bbox.lo = glm::min(groupBbox.lo, geometry.bbox.lo);
    geometry.bbox.hi = glm::max(groupBbox.hi, geometry.bbox.hi);

    if(context->lodLevel != group.lodLevel)
    {
      context->lodLevel                  = group.lodLevel;
      const shaderio::LodLevel* previous = group.lodLevel ? &geometry.lodLevels[group.lodLevel - 1] : nullptr;

      shaderio::LodLevel initLevel{};
      initLevel.clusterOffset = previous ? previous->clusterOffset + previous->clusterCount : 0;
      initLevel.groupOffset   = previous ? previous->groupOffset + previous->groupCount : 0;

      initLevel.minBoundingSphereRadius = FLT_MAX;
      initLevel.minMaxQuadricError      = FLT_MAX;

      // add new
      geometry.lodLevels.push_back(initLevel);
    }

    geometry.lodLevels[level].clusterCount += groupInfo.clusterCount;
    geometry.lodLevels[level].groupCount++;

    // USE_BLAS_SHARING
    //
    // For the BLAS sharing technique we need to figure out the conservative
    // lod range that an instance may cover. We store for each lod level
    // the smallest possible group bounding sphere as well as the smallest
    // maximum error found in any group.

    // The technique will use these values to artificially place a lod sphere
    // at the far end of an instance and evaluate its lod metric. The minima
    // ensure that there can't be any group in the instance's sphere that would
    // behave such a way that it requires lower detail.
    //
    // See `instance_classify_lod.comp.glsl` shader.

    geometry.lodLevels[level].minBoundingSphereRadius =
        std::min(geometry.lodLevels[level].minBoundingSphereRadius, group.traversalMetric.boundingSphereRadius);
    geometry.lodLevels[level].minMaxQuadricError =
        std::min(geometry.lodLevels[level].minMaxQuadricError, group.traversalMetric.maxQuadricError);

    // stats

    geometry.clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, geometry.clusterMaxTrianglesCount);
    geometry.clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, geometry.clusterMaxVerticesCount);

    if(level == 0)
    {
      geometry.hiClustersCount += groupInfo.clusterCount;
      geometry.hiTriangleCount += groupInfo.triangleCount;
      geometry.hiVerticesCount += groupInfo.vertexCount;
    }

    geometry.totalClustersCount += groupInfo.clusterCount;
    geometry.totalTriangleCount += groupInfo.triangleCount;
    geometry.totalVerticesCount += groupInfo.vertexCount;

    // primary allocation and export
    if(useOrderedLock)
    {
      // groupInfo.offsetBytes was acquired in an orderly fashion
      if(geometry.groupData.size() < groupInfo.offsetBytes + groupInfo.sizeBytes)
      {
        geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);
      }
    }
    else
    {
      // without inner parallelism we get called linearly anyway
      groupInfo.offsetBytes = geometry.groupData.size();
      geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);

      // may also need to generate the groupIndex manually
      if(groupIndex == ~0)
      {
        groupIndex = uint32_t(geometry.groupInfos.size());
        geometry.groupInfos.resize(groupIndex + 1);
      }
    }

    geometry.groupInfos[groupIndex] = groupInfo;

    {
      GroupStorage groupStorage(&geometry.groupData[groupInfo.offsetBytes], groupInfo);

      size_t startAddress = size_t(groupStorage.group);

      // always zero, patched in during upload
      groupStorage.group->residentID        = 0;
      groupStorage.group->clusterResidentID = 0;

      // regular values
      groupStorage.group->lodLevel        = level;
      groupStorage.group->clusterCount    = groupInfo.clusterCount;
      groupStorage.group->traversalMetric = group.traversalMetric;

      memcpy(groupStorage.clusters.data(), groupTempStorage.clusters.data(), groupStorage.clusters.size_bytes());

      // patch adjustments
      for(uint32_t c = 0; c < clusterCount; c++)
      {
        shaderio::Cluster& groupCluster = groupStorage.clusters[c];
        groupCluster.vertices = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + groupCluster.vertices);
        groupCluster.indices = groupStorage.getClusterLocalOffset(c, groupStorage.indices.data() + groupCluster.indices);
      }
      memcpy(groupStorage.clusterGeneratingGroups.data(), groupTempStorage.clusterGeneratingGroups.data(),
             groupStorage.clusterGeneratingGroups.size_bytes());
      // fill padded array with zero
      {
        size_t padSize =
            size_t(groupStorage.clusterBboxes.data()) - size_t(groupStorage.clusterGeneratingGroups.data() + clusterCount);
        if(padSize)
        {
          memset(groupStorage.clusterGeneratingGroups.data() + clusterCount, 0, padSize);
        }
      }

      memcpy(groupStorage.clusterBboxes.data(), groupTempStorage.clusterBboxes.data(), groupStorage.clusterBboxes.size_bytes());
      memcpy(groupStorage.vertices.data(), groupTempStorage.vertices.data(), groupStorage.vertices.size_bytes());
      memcpy(groupStorage.indices.data(), groupTempStorage.indices.data(), groupStorage.indices.size_bytes());
      // fill padded array with zeros
      {
        size_t padSize = size_t(groupStorage.raw + groupInfo.sizeBytes)
                         - size_t(groupStorage.indices.data() + groupInfo.triangleCount * 3);
        if(padSize)
        {
          memset(groupStorage.indices.data() + groupInfo.triangleCount * 3, 0, padSize);
        }
      }
    }
  }

  return groupIndex;
}

// Callback used by the mesoptimizer's clusterlod generator. Run once
// for each lod level (except the very last). task_count is the number
// of groups to be processed within this lod level.
// This sample only uses this callback when we intend to multi-thread within
// a single geometry. When loading scenes with many objects this is less likely
// to be used.
void Scene::clodIterationMeshoptimizer(void* intermediate_context, void* output_context, int depth, size_t task_count)
{
  TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
  GeometryStorage& geometry = context->geometry;

  context->levelGroupOffset      = geometry.groupInfos.size();
  context->levelGroupOffsetValid = true;
  geometry.groupInfos.resize(context->levelGroupOffset + task_count);


  nvutils::parallel_batches_pooled<1>(
      task_count,
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        clodBuild_iterationTask(intermediate_context, output_context, idx, threadInnerIdx);
      },
      context->processingInfo.numInnerThreads);


  context->levelGroupOffsetValid = false;
}

// callback used by mesoptimizer's clusterlod generator to provide the
// result cluster group for further processing.
int Scene::clodGroupMeshoptimizer(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, uint32_t thread_index)
{
  TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
  GeometryStorage& geometry = context->geometry;

  TempGroup tempGroup;
  tempGroup.lodLevel                             = group.depth;
  tempGroup.clusterCount                         = uint32_t(cluster_count);
  tempGroup.traversalMetric.boundingSphereX      = group.simplified.center[0];
  tempGroup.traversalMetric.boundingSphereY      = group.simplified.center[1];
  tempGroup.traversalMetric.boundingSphereZ      = group.simplified.center[2];
  tempGroup.traversalMetric.boundingSphereRadius = group.simplified.radius;
  tempGroup.traversalMetric.maxQuadricError      = group.simplified.error;

  auto fnClusterProvider = [&clusters](uint32_t c) {
    TempCluster cluster;
    cluster.generatingGroup = clusters[c].refined;
    cluster.indexCount      = uint32_t(clusters[c].index_count);
    cluster.indices         = clusters[c].indices;
    return cluster;
  };

  // we test against `context->levelGroupOffsetValid` because this function is also called for
  // the last lod-level without being wrapped by `clodIterationMeshoptimizer`, which does do the setup
  // of `context->levelGroupOffset`

  uint32_t groupIndex =
      context->innerThreadingActive && context->levelGroupOffsetValid ? uint32_t(context->levelGroupOffset + task_index) : ~0u;

  return storeGroup(context, thread_index, groupIndex, tempGroup, fnClusterProvider);
}

void Scene::buildGeometryClusterLodMeshoptimizer(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  clodConfig clodInfo = m_config.meshoptPreferRayTracing ? clodDefaultConfigRT(m_config.clusterTriangles) :
                                                           clodDefaultConfig(m_config.clusterTriangles);

  clodInfo.cluster_fill_weight  = m_config.meshoptFillWeight;
  clodInfo.cluster_split_factor = m_config.meshoptSplitFactor;
  clodInfo.max_vertices         = m_config.clusterVertices;
  clodInfo.partition_size       = m_config.clusterGroupSize;
  clodInfo.partition_spatial    = true;
  clodInfo.partition_sort       = true;

  // this only reorders triangles within cluster
  clodInfo.optimize_clusters = true;

  // account for meshopt_partitionClusters's using a target value with a higher worst case
  while((clodInfo.partition_size + clodInfo.partition_size / 3) > m_config.clusterGroupSize)
  {
    clodInfo.partition_size--;
  }

  // These control the error propagation across lod levels to
  // account for simplifying an already simplified mesh.
  clodInfo.simplify_error_merge_previous = m_config.lodErrorMergePrevious;
  clodInfo.simplify_error_merge_additive = m_config.lodErrorMergeAdditive;

  clodMesh inputMesh                = {};
  inputMesh.vertex_positions        = reinterpret_cast<const float*>(geometry.vertexPositions.data());
  inputMesh.vertex_count            = geometry.vertexPositions.size();
  inputMesh.vertex_positions_stride = sizeof(glm::vec3);
  inputMesh.index_count             = geometry.triangles.size() * 3;
  inputMesh.indices                 = reinterpret_cast<const uint32_t*>(geometry.triangles.data());

  float attributeWeights[9] = {};

  if(geometry.attributesWithWeights)
  {
    if(m_config.simplifyNormalWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      attributeWeights[geometry.attributeNormalOffset + 0] = m_config.simplifyNormalWeight;
      attributeWeights[geometry.attributeNormalOffset + 1] = m_config.simplifyNormalWeight;
      attributeWeights[geometry.attributeNormalOffset + 2] = m_config.simplifyNormalWeight;
    }
    if(m_config.simplifyUvWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_UV))
    {
      attributeWeights[geometry.attributeUvOffset + 0] = m_config.simplifyUvWeight;
      attributeWeights[geometry.attributeUvOffset + 1] = m_config.simplifyUvWeight;
    }
    if(m_config.simplifyTangentWeight > 0 && m_config.simplifyTangentSignWeight > 0
       && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
    {
      attributeWeights[geometry.attributeTangentOffset + 0] = m_config.simplifyTangentWeight;
      attributeWeights[geometry.attributeTangentOffset + 1] = m_config.simplifyTangentWeight;
      attributeWeights[geometry.attributeTangentOffset + 2] = m_config.simplifyTangentWeight;
      attributeWeights[geometry.attributeTangentOffset + 3] = m_config.simplifyTangentSignWeight;
    }
    // TODO  material index handling...

    inputMesh.attribute_count          = geometry.attributesWithWeights;
    inputMesh.vertex_attributes        = geometry.vertexAttributes.data();
    inputMesh.vertex_attributes_stride = sizeof(float) * inputMesh.attribute_count;
    inputMesh.attribute_weights        = attributeWeights;
  }

  TempContext context = {processingInfo, geometry};

  GroupInfo worstGroup       = {};
  worstGroup.clusterCount    = uint8_t(m_config.clusterGroupSize);
  worstGroup.vertexCount     = uint16_t(m_config.clusterGroupSize * m_config.clusterVertices);
  worstGroup.triangleCount   = uint16_t(m_config.clusterGroupSize * m_config.clusterTriangles);
  worstGroup.attributeBits   = geometry.attributeBits;
  worstGroup.vertexDataCount = worstGroup.estimateVertexDataCount();
  worstGroup.sizeBytes       = worstGroup.computeSize();

  context.innerThreadingActive   = processingInfo.numInnerThreads > 1;
  context.threadGroupInfo        = worstGroup;
  context.threadGroupStorageSize = uint32_t(worstGroup.computeSize());
  context.threadGroupSize        = nvutils::align_up(context.threadGroupStorageSize, 4) + sizeof(uint32_t) * 256 * 3;
  context.threadGroupDatas.resize(context.threadGroupSize * processingInfo.numInnerThreads);

  size_t reservedClusters  = (geometry.triangles.size() + m_config.clusterTriangles - 1) / m_config.clusterTriangles;
  size_t reservedGroups    = (reservedClusters + m_config.clusterGroupSize - 1) / m_config.clusterGroupSize;
  size_t reservedTriangles = geometry.triangles.size();

  reservedClusters  = size_t(double(reservedClusters) * 2.0);
  reservedGroups    = size_t(double(reservedGroups) * 3.0);
  reservedTriangles = size_t(double(reservedTriangles) * 2.0);

  size_t reservedData = 0;
  reservedData += sizeof(shaderio::Group) * reservedGroups;
  reservedData += sizeof(shaderio::Cluster) * reservedClusters;
  reservedData += sizeof(shaderio::BBox) * reservedClusters;
  reservedData += sizeof(uint32_t) * reservedClusters;
  reservedData += sizeof(uint8_t) * reservedTriangles;
  reservedData += sizeof(glm::vec3) * reservedClusters * m_config.clusterVertices;

  geometry.groupData.reserve(reservedData);
  geometry.groupInfos.reserve(reservedGroups);
  geometry.lodLevels.reserve(32);

  clodBuild(clodInfo, inputMesh, &context, clodGroupMeshoptimizer,
            processingInfo.numInnerThreads > 1 ? clodIterationMeshoptimizer : nullptr);

  // can nuke inputs
  geometry.triangles        = {};
  geometry.vertexPositions  = {};
  geometry.vertexAttributes = {};

  // check last lod level
  geometry.lodLevelsCount = uint32_t(geometry.lodLevels.size());
  if(geometry.lodLevelsCount)
  {
    shaderio::LodLevel& lastLodLevel = geometry.lodLevels.back();

    if(lastLodLevel.groupCount != 1 || lastLodLevel.clusterCount != 1)
    {
      assert(0);
      LOGE("clodBuild failed: last lod level has more than one cluster\n");
      std::exit(-1);
    }
  }

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.groupInfos.shrink_to_fit();
  geometry.groupData.shrink_to_fit();
  geometry.lodLevels.shrink_to_fit();

  buildGeometryLodHierarchyMeshoptimizer(processingInfo, geometry);
}

void Scene::buildGeometryLodHierarchyMeshoptimizer(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  // for each lod level build hierarchy

  uint32_t lodLevelCount = geometry.lodLevelsCount;

  std::vector<Range> lodNodeRanges(lodLevelCount);

  // top root is first
  // lod-level many lod-roots next
  // then rest
  {
    uint32_t nodeOffset = 1 + lodLevelCount;

    for(uint32_t lodLevel = 0; lodLevel < lodLevelCount; lodLevel++)
    {
      const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];

      // groups as leaves
      uint32_t nodeCount = lodLevelInfo.groupCount;

      // then nodes on top
      uint32_t iterationCount = nodeCount;

      while(iterationCount > 1)
      {
        iterationCount = (iterationCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
        nodeCount += iterationCount;
      }

      // subtract root, already accounted for
      nodeCount--;

      lodNodeRanges[lodLevel].offset = nodeOffset;
      lodNodeRanges[lodLevel].count  = nodeCount;
      nodeOffset += nodeCount;
    }
    geometry.lodNodes.resize(nodeOffset);
  }

  // build per-level trees
  nvutils::parallel_batches_pooled<1>(
      lodLevelCount,
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        uint32_t                  lodLevel     = uint32_t(idx);
        const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];
        const Range&              lodNodeRange = lodNodeRanges[lodLevel];

        // groups as leaves
        uint32_t nodeCount      = lodLevelInfo.groupCount;
        uint32_t nodeOffset     = lodNodeRange.offset;
        uint32_t lastNodeOffset = nodeOffset;

        for(uint32_t g = 0; g < nodeCount; g++)
        {
          uint32_t         groupID   = g + lodLevelInfo.groupOffset;
          const GroupInfo& groupInfo = geometry.groupInfos[groupID];
          GroupView        groupView(geometry.groupData, groupInfo);

          shaderio::Node& node = nodeCount == 1 ? geometry.lodNodes[1 + lodLevel] : geometry.lodNodes[nodeOffset++];

          node                                      = {};
          node.groupRange.isGroup                   = 1;
          node.groupRange.groupIndex                = groupID;
          node.groupRange.groupClusterCountMinusOne = groupInfo.clusterCount - 1;
          node.traversalMetric                      = groupView.group->traversalMetric;
        }
        // special case single node, directly stored to root section
        if(nodeCount == 1)
        {
          nodeOffset++;
        }

        // then nodes on top
        uint32_t depth          = 0;
        uint32_t iterationCount = nodeCount;

        std::vector<uint32_t>       partitionedIndices;
        std::vector<shaderio::Node> oldNodes;

        while(iterationCount > 1)
        {
          uint32_t        lastNodeCount = iterationCount;
          shaderio::Node* lastNodes     = &geometry.lodNodes[lastNodeOffset];

          // partition last nodes into children for new nodes
          partitionedIndices.resize(lastNodeCount);
          meshopt_spatialClusterPoints(partitionedIndices.data(), &lastNodes->traversalMetric.boundingSphereX,
                                       lastNodeCount, sizeof(shaderio::Node), m_config.preferredNodeWidth);

          {
            // re-order last nodes by new partition
            oldNodes.clear();
            oldNodes.insert(oldNodes.end(), lastNodes, lastNodes + lastNodeCount);

            for(uint32_t n = 0; n < lastNodeCount; n++)
            {
              lastNodes[n] = oldNodes[partitionedIndices[n]];
            }
          }

          // number of new nodes
          iterationCount = (lastNodeCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;

          // root is stored at special place
          shaderio::Node* newNodes = iterationCount == 1 ? &geometry.lodNodes[1 + lodLevel] : &geometry.lodNodes[nodeOffset];

          for(uint32_t n = 0; n < iterationCount; n++)
          {
            shaderio::Node& node          = newNodes[n];
            shaderio::Node* childrenNodes = &lastNodes[n * m_config.preferredNodeWidth];

            uint32_t childCount = std::min((n + 1) * m_config.preferredNodeWidth, lastNodeCount) - n * m_config.preferredNodeWidth;

            node                                 = {};
            node.nodeRange.isGroup               = 0;
            node.nodeRange.childCountMinusOne    = childCount - 1;
            node.nodeRange.childOffset           = lastNodeOffset + n * m_config.preferredNodeWidth;
            node.traversalMetric.maxQuadricError = 0;

            for(uint32_t c = 0; c < childCount; c++)
            {
              node.traversalMetric.maxQuadricError =
                  std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
            }

            meshopt_Bounds merged =
                meshopt_computeSphereBounds(&childrenNodes[0].traversalMetric.boundingSphereX, childCount, sizeof(shaderio::Node),
                                            &childrenNodes[0].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));

            node.traversalMetric.boundingSphereX      = merged.center[0];
            node.traversalMetric.boundingSphereY      = merged.center[1];
            node.traversalMetric.boundingSphereZ      = merged.center[2];
            node.traversalMetric.boundingSphereRadius = merged.radius;
          }

          lastNodeOffset = nodeOffset;
          nodeOffset += iterationCount;
          depth++;
        }

        nodeOffset--;
        assert(lodNodeRange.offset + lodNodeRange.count == nodeOffset);
      },
      processingInfo.numInnerThreads);

  // then setup top tree root
  {
    meshopt_Bounds merged =
        meshopt_computeSphereBounds(&geometry.lodNodes[1].traversalMetric.boundingSphereX, lodLevelCount, sizeof(shaderio::Node),
                                    &geometry.lodNodes[1].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));

    shaderio::Node& node          = geometry.lodNodes[0];
    shaderio::Node* childrenNodes = &geometry.lodNodes[1];

    node                                      = {};
    node.nodeRange.isGroup                    = 0;
    node.nodeRange.childCountMinusOne         = lodLevelCount - 1;
    node.nodeRange.childOffset                = 1;
    node.traversalMetric.boundingSphereX      = merged.center[0];
    node.traversalMetric.boundingSphereY      = merged.center[1];
    node.traversalMetric.boundingSphereZ      = merged.center[2];
    node.traversalMetric.boundingSphereRadius = merged.radius;
    node.traversalMetric.maxQuadricError      = 0;

    for(uint32_t c = 0; c < lodLevelCount; c++)
    {
      node.traversalMetric.maxQuadricError =
          std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
    }
  }
}

void Scene::buildGeometryClusterLodNvLib(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  // WARNING: to be removed / deprecated
  // NVIDIA will deprecate the cluster lod builder library in favor of recommending the use of meshoptimizer.

  nvclusterlod_Result result;

  nvclusterlod_MeshInput lodMeshInput;
  lodMeshInput.decimationFactor = m_config.lodLevelDecimationFactor;
  lodMeshInput.triangleCount    = uint32_t(geometry.triangles.size());
  lodMeshInput.triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(geometry.triangles.data());
  lodMeshInput.vertexCount      = uint32_t(geometry.vertexPositions.size());
  lodMeshInput.vertexStride     = sizeof(glm::vec4);
  lodMeshInput.vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(geometry.vertexPositions.data());

  lodMeshInput.clusterConfig.minClusterSize = (m_config.clusterTriangles * 3) / 4;
  lodMeshInput.clusterConfig.maxClusterSize = m_config.clusterTriangles;

  if(m_config.clusterVertices < m_config.clusterTriangles * 3)
  {
    lodMeshInput.clusterConfig.maxClusterVertices = m_config.clusterVertices;
    lodMeshInput.clusterConfig.itemVertexCount    = 3;
  }

  lodMeshInput.groupConfig.minClusterSize = (m_config.clusterGroupSize * 3) / 4;
  lodMeshInput.groupConfig.maxClusterSize = m_config.clusterGroupSize;

  TempContext temp = {processingInfo, geometry};
  result           = nvclusterlod::generateLodMesh(processingInfo.lodContext, lodMeshInput, temp.lodMesh);
  if(result != NVCLUSTERLOD_SUCCESS)
  {
    assert(0);
    LOGE("nvclusterlod::generateLodMesh failed: %d\n", result);
    std::exit(-1);
  }

  // can remove original triangle indices
  geometry.triangles = {};

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  temp.lodMesh.triangleVertices.shrink_to_fit();
  temp.lodMesh.clusterTriangleRanges.shrink_to_fit();
  temp.lodMesh.clusterGeneratingGroups.shrink_to_fit();
  temp.lodMesh.clusterBoundingSpheres.shrink_to_fit();
  temp.lodMesh.groupQuadricErrors.shrink_to_fit();
  temp.lodMesh.groupClusterRanges.shrink_to_fit();
  temp.lodMesh.lodLevelGroupRanges.shrink_to_fit();


  nvclusterlod_HierarchyInput hierarchyInput = {};
  hierarchyInput.clusterCount                = uint32_t(temp.lodMesh.clusterBoundingSpheres.size());
  hierarchyInput.clusterBoundingSpheres      = temp.lodMesh.clusterBoundingSpheres.data();
  hierarchyInput.clusterGeneratingGroups     = temp.lodMesh.clusterGeneratingGroups.data();
  hierarchyInput.groupClusterRanges          = temp.lodMesh.groupClusterRanges.data();
  hierarchyInput.groupQuadricErrors          = temp.lodMesh.groupQuadricErrors.data();
  hierarchyInput.lodLevelGroupRanges         = temp.lodMesh.lodLevelGroupRanges.data();
  hierarchyInput.lodLevelCount               = uint32_t(temp.lodMesh.lodLevelGroupRanges.size());
  hierarchyInput.groupCount                  = int32_t(temp.lodMesh.groupClusterRanges.size());

  // required to later traverse the lod hierarchy in parallel
  // Note we build hierarchies over each lod level's groups
  // and then join them together to a single hierarchy.
  // This is key to the parallel traversal algorithm that traverse multiple lod levels
  // at once.

  result = nvclusterlod::generateLodHierarchy(processingInfo.lodContext, hierarchyInput, temp.lodHierarchy);
  if(result != NVCLUSTERLOD_SUCCESS)
  {
    assert(0);
    LOGE("nvclusterlod::generateLodHierarchy failed: %d\n", result);
    std::exit(-1);
  }

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  temp.lodHierarchy.nodes.shrink_to_fit();
  temp.lodHierarchy.groupCumulativeBoundingSpheres.shrink_to_fit();
  temp.lodHierarchy.groupCumulativeQuadricError.shrink_to_fit();

  // these are compatible
  std::vector<nvclusterlod_HierarchyNode>& nodes = (std::vector<nvclusterlod_HierarchyNode>&)geometry.lodNodes;
  nodes                                          = std::move(temp.lodHierarchy.nodes);

  // no longer needed
  temp.lodMesh.groupQuadricErrors = {};

  geometry.lodLevelsCount = uint32_t(temp.lodMesh.lodLevelGroupRanges.size());
  geometry.lodLevels.reserve(temp.lodMesh.lodLevelGroupRanges.size());
  temp.groupLodLevels.resize(temp.lodMesh.groupClusterRanges.size());

  for(size_t level = 0; level < temp.lodMesh.lodLevelGroupRanges.size(); level++)
  {
    nvcluster_Range groupRange = temp.lodMesh.lodLevelGroupRanges[level];

    for(size_t g = groupRange.offset; g < groupRange.offset + groupRange.count; g++)
    {
      temp.groupLodLevels[g] = uint8_t(level);
    }
  }

  GroupInfo worstGroup       = {};
  worstGroup.clusterCount    = uint8_t(m_config.clusterGroupSize);
  worstGroup.vertexCount     = uint16_t(m_config.clusterGroupSize * m_config.clusterVertices);
  worstGroup.triangleCount   = uint16_t(m_config.clusterGroupSize * m_config.clusterTriangles);
  worstGroup.attributeBits   = geometry.attributeBits;
  worstGroup.vertexDataCount = worstGroup.estimateVertexDataCount();
  worstGroup.sizeBytes       = worstGroup.computeSize();

  temp.innerThreadingActive   = processingInfo.numInnerThreads > 1;
  temp.threadGroupInfo        = worstGroup;
  temp.threadGroupStorageSize = uint32_t(worstGroup.computeSize());
  temp.threadGroupSize        = nvutils::align_up(temp.threadGroupStorageSize, 4) + sizeof(uint32_t) * 256 * 3;
  temp.threadGroupDatas.resize(temp.threadGroupSize * processingInfo.numInnerThreads);

  geometry.groupInfos.resize(temp.lodMesh.groupClusterRanges.size());

  // process groups in parallel
  nvutils::parallel_batches_pooled(
      temp.lodMesh.groupClusterRanges.size(),
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        TempGroup group;
        group.clusterCount = temp.lodMesh.groupClusterRanges[idx].count;
        group.lodLevel     = temp.groupLodLevels[idx];

        nvclusterlod_Sphere sphere                 = temp.lodHierarchy.groupCumulativeBoundingSpheres[idx];
        group.traversalMetric.boundingSphereX      = sphere.center.x;
        group.traversalMetric.boundingSphereY      = sphere.center.y;
        group.traversalMetric.boundingSphereZ      = sphere.center.z;
        group.traversalMetric.boundingSphereRadius = sphere.radius;
        group.traversalMetric.maxQuadricError      = temp.lodHierarchy.groupCumulativeQuadricError[idx];

        nvcluster_Range clusterRange = temp.lodMesh.groupClusterRanges[idx];

        auto fnClusterProvider = [&](uint32_t c) {
          TempCluster     cluster;
          uint32_t        clusterIndex  = clusterRange.offset + c;
          nvcluster_Range triangleRange = temp.lodMesh.clusterTriangleRanges[clusterIndex];

          cluster.generatingGroup = temp.lodMesh.clusterGeneratingGroups[clusterIndex];
          cluster.indexCount      = triangleRange.count * 3;
          cluster.indices         = &temp.lodMesh.triangleVertices[triangleRange.offset].x;
          return cluster;
        };

        storeGroup(&temp, threadInnerIdx, uint32_t(idx), group, fnClusterProvider);
      },
      processingInfo.numInnerThreads);

  geometry.vertexPositions  = {};
  geometry.vertexAttributes = {};
}

}  // namespace lodclusters