
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

#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <meshoptimizer.h>

#include "scene.hpp"

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

  // build localized index buffers

  geometry.localTriangles.resize(geometry.lodMesh.triangleVertices.size() * 3);
  geometry.localVertices.resize(geometry.lodMesh.triangleVertices.size() * 3);
  geometry.clusterVertexRanges.resize(geometry.lodMesh.clusterTriangleRanges.size());

  std::vector<uint32_t> threadClusterMaxVertices(processingInfo.numInnerThreads, 0);
  std::vector<uint32_t> threadClusterMaxTriangles(processingInfo.numInnerThreads, 0);
  std::vector<uint32_t> threadCacheEarly(processingInfo.numInnerThreads * 256 * 2);

  nvutils::parallel_batches_pooled(
      geometry.lodMesh.clusterTriangleRanges.size(),
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        nvcluster_Range& vertexRange             = geometry.clusterVertexRanges[idx];
        nvcluster_Range  triangleRange           = geometry.lodMesh.clusterTriangleRanges[idx];
        const uint32_t* __restrict inputTriangle = &geometry.lodMesh.triangleVertices[triangleRange.offset].x;
        uint8_t* __restrict outputTriangle       = &geometry.localTriangles[triangleRange.offset * 3];
        uint32_t* __restrict outVertices         = &geometry.localVertices[triangleRange.offset * 3];
        uint32_t* vertexCacheEarlyValue          = &threadCacheEarly[threadInnerIdx * 256 * 2];
        uint32_t* vertexCacheEarlyPos            = vertexCacheEarlyValue + 256;
        memset(vertexCacheEarlyValue, ~0, sizeof(uint32_t) * 256);

        uint32_t count = 0;

        for(uint32_t i = 0; i < triangleRange.count * 3; i++)
        {
          uint32_t vertexIndex = inputTriangle[i];
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
            for(uint32_t v = 0; v < count; v++)
            {
              if(outVertices[v] == vertexIndex)
              {
                cacheIndex = v;
              }
            }
          }

          if(cacheIndex == ~0)
          {
            cacheIndex                                = count++;
            outVertices[cacheIndex]                   = vertexIndex;
            vertexCacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;
            vertexCacheEarlyPos[vertexIndex & 0xFF]   = cacheIndex;
          }
          outputTriangle[i] = uint8_t(cacheIndex);
        }

        vertexRange.count = count;
        threadClusterMaxVertices[threadInnerIdx] = std::max(threadClusterMaxVertices[threadInnerIdx], vertexRange.count);
        threadClusterMaxTriangles[threadInnerIdx] = std::max(threadClusterMaxTriangles[threadInnerIdx], triangleRange.count);

        assert(triangleRange.count <= m_config.clusterTriangles);
        assert(vertexRange.count <= m_config.clusterVertices);
      },
      processingInfo.numInnerThreads);

  // no longer needed
  geometry.lodMesh.triangleVertices = {};

  geometry.clusterMaxVerticesCount  = 0;
  geometry.clusterMaxTrianglesCount = 0;
  for(uint32_t t = 0; t < processingInfo.numInnerThreads; t++)
  {
    geometry.clusterMaxVerticesCount  = std::max(geometry.clusterMaxVerticesCount, threadClusterMaxVertices[t]);
    geometry.clusterMaxTrianglesCount = std::max(geometry.clusterMaxTrianglesCount, threadClusterMaxTriangles[t]);
  }

  // compaction pass
  uint32_t* localVertices = geometry.localVertices.data();

  uint32_t offset = 0;
  for(size_t c = 0; c < geometry.lodMesh.clusterTriangleRanges.size(); c++)
  {
    nvcluster_Range& vertexRange   = geometry.clusterVertexRanges[c];
    nvcluster_Range  triangleRange = geometry.lodMesh.clusterTriangleRanges[c];

    vertexRange.offset = offset;

    memmove(&localVertices[offset], &localVertices[triangleRange.offset * 3], sizeof(uint32_t) * vertexRange.count);

    offset += vertexRange.count;
  }

  geometry.localVertices.resize(offset);
  geometry.localVertices.shrink_to_fit();

  geometry.hiClustersCount = 0;
  geometry.hiTriangleCount = 0;
  geometry.hiVerticesCount = 0;

  {
    nvcluster_Range groupRange = geometry.lodMesh.lodLevelGroupRanges[0];

    for(size_t g = groupRange.offset; g < groupRange.offset + groupRange.count; g++)
    {
      nvcluster_Range clusterRange = geometry.lodMesh.groupClusterRanges[g];

      uint32_t lastCluster  = clusterRange.offset + clusterRange.count - 1;
      uint32_t firstCluster = clusterRange.offset;

      geometry.hiClustersCount += clusterRange.count;
      geometry.hiTriangleCount += geometry.lodMesh.clusterTriangleRanges[lastCluster].count
                                  + geometry.lodMesh.clusterTriangleRanges[lastCluster].offset
                                  - geometry.lodMesh.clusterTriangleRanges[firstCluster].offset;
      geometry.hiVerticesCount += geometry.clusterVertexRanges[lastCluster].count
                                  + geometry.clusterVertexRanges[lastCluster].offset
                                  - geometry.clusterVertexRanges[firstCluster].offset;
    }
  }

  geometry.totalTriangleCount = uint32_t(geometry.localTriangles.size() / 3);
  geometry.totalVerticesCount = uint32_t(geometry.localVertices.size());
  geometry.totalClustersCount = uint32_t(geometry.lodMesh.clusterTriangleRanges.size());
}

void Scene::clodIterationMeshoptimizer(void* intermediate_context, void* output_context, int depth, size_t task_count)
{
  MeshoptContext*  context  = reinterpret_cast<MeshoptContext*>(output_context);
  GeometryStorage& geometry = context->geometry;

  nvutils::parallel_batches_pooled<1>(
      task_count,
      [&](uint64_t idx, uint32_t threadInnerIdx) { clodBuild_iterationTask(intermediate_context, output_context, idx); },
      context->processingInfo.numInnerThreads);
}

int Scene::clodGroupMeshoptimizer(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count)
{
  MeshoptContext*  context  = reinterpret_cast<MeshoptContext*>(output_context);
  GeometryStorage& geometry = context->geometry;

  std::lock_guard lock(context->groupMutex);

  uint32_t level = uint32_t(group.depth);

  if(context->depth != group.depth)
  {
    context->depth = group.depth;

    nvcluster_Range groupRange;
    groupRange.offset = uint32_t(geometry.lodMesh.groupClusterRanges.size());

    // patch last lod range
    if(group.depth > 0)
    {
      uint32_t groupCount = groupRange.offset - geometry.lodMesh.lodLevelGroupRanges.back().offset;

      geometry.lodMesh.lodLevelGroupRanges.back().count = groupCount;
      geometry.lodLevels.back().groupCount              = groupCount;
    }

    // add new
    geometry.lodMesh.lodLevelGroupRanges.push_back(groupRange);

    shaderio::LodLevel initLevel{};
    initLevel.groupOffset             = groupRange.offset;
    initLevel.minBoundingSphereRadius = FLT_MAX;
    initLevel.minMaxQuadricError      = FLT_MAX;
    geometry.lodLevels.push_back(initLevel);
  }

  size_t groupIndex = geometry.lodMesh.groupClusterRanges.size();
  if(groupIndex >= geometry.lodMesh.groupClusterRanges.size())
  {
    geometry.groupLodLevels.resize(groupIndex + 1);
    geometry.lodMesh.groupClusterRanges.resize(groupIndex + 1);
    geometry.lodHierarchy.groupCumulativeBoundingSpheres.resize(groupIndex + 1);
    geometry.lodHierarchy.groupCumulativeQuadricError.resize(groupIndex + 1);
  }

  size_t cluster_index = geometry.lodMesh.clusterTriangleRanges.size();

  nvcluster_Range clusterRange;
  clusterRange.offset = uint32_t(cluster_index);
  clusterRange.count  = uint32_t(cluster_count);

  geometry.groupLodLevels[groupIndex]             = uint8_t(level);
  geometry.lodMesh.groupClusterRanges[groupIndex] = clusterRange;

  geometry.lodHierarchy.groupCumulativeBoundingSpheres[groupIndex] = {
      {group.simplified.center[0], group.simplified.center[1], group.simplified.center[2]}, group.simplified.radius};
  geometry.lodHierarchy.groupCumulativeQuadricError[groupIndex] = group.simplified.error;

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
      std::min(geometry.lodLevels[level].minBoundingSphereRadius, group.simplified.radius);
  geometry.lodLevels[level].minMaxQuadricError = std::min(geometry.lodLevels[level].minMaxQuadricError, group.simplified.error);


  // add clusters

  geometry.lodMesh.clusterTriangleRanges.resize(cluster_index + cluster_count);
  geometry.lodMesh.clusterBoundingSpheres.resize(cluster_index + cluster_count);
  geometry.lodMesh.clusterGeneratingGroups.resize(cluster_index + cluster_count);

  for(size_t c = 0; c < cluster_count; c++, cluster_index++)
  {
    const clodCluster cluster = clusters[c];

    nvcluster_Range triangleRange;
    triangleRange.offset = uint32_t(context->geometry.lodMesh.triangleVertices.size());
    triangleRange.count  = uint32_t(cluster.index_count / 3);

    geometry.lodMesh.clusterTriangleRanges[cluster_index] = triangleRange;
    geometry.lodMesh.triangleVertices.insert(geometry.lodMesh.triangleVertices.end(), (const nvclusterlod_Vec3u*)cluster.indices,
                                             (const nvclusterlod_Vec3u*)(cluster.indices + cluster.index_count));

    geometry.lodMesh.clusterBoundingSpheres[cluster_index] = {
        {cluster.bounds.center[0], cluster.bounds.center[1], cluster.bounds.center[2]}, cluster.bounds.radius};
    geometry.lodMesh.clusterGeneratingGroups[cluster_index] = cluster.refined;
  }

  return int(groupIndex);
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

  // this only reorders triangles within cluster, run it, if we don't do triangle strips
  clodInfo.optimize_raster = !m_config.clusterStripify;

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
  inputMesh.vertex_positions        = reinterpret_cast<const float*>(geometry.vertices.data());
  inputMesh.vertex_count            = geometry.vertices.size();
  inputMesh.vertex_positions_stride = sizeof(glm::vec4);
  inputMesh.index_count             = geometry.globalTriangles.size() * 3;
  inputMesh.indices                 = reinterpret_cast<const uint32_t*>(geometry.globalTriangles.data());

  MeshoptContext context = {processingInfo, geometry};

  size_t reservedClusters = (geometry.globalTriangles.size() + m_config.clusterTriangles - 1) / m_config.clusterTriangles;
  size_t reservedGroups    = (reservedClusters + m_config.clusterGroupSize - 1) / m_config.clusterGroupSize;
  size_t reservedTriangles = geometry.globalTriangles.size();

  reservedClusters  = size_t(double(reservedClusters) * 2.0);
  reservedGroups    = size_t(double(reservedGroups) * 3.0);
  reservedTriangles = size_t(double(reservedTriangles) * 2.0);

  geometry.lodMesh.groupClusterRanges.reserve(reservedGroups);
  geometry.groupLodLevels.reserve(reservedGroups);

  geometry.lodHierarchy.groupCumulativeBoundingSpheres.reserve(reservedGroups);
  geometry.lodHierarchy.groupCumulativeQuadricError.reserve(reservedGroups);

  geometry.lodMesh.clusterBoundingSpheres.reserve(reservedClusters);
  geometry.lodMesh.clusterGeneratingGroups.reserve(reservedClusters);
  geometry.lodMesh.clusterTriangleRanges.reserve(reservedClusters);

  geometry.lodMesh.triangleVertices.reserve(reservedTriangles);
  geometry.lodMesh.lodLevelGroupRanges.reserve(32);
  geometry.lodLevels.reserve(32);

  clodBuild(clodInfo, inputMesh, &context, clodGroupMeshoptimizer,
            processingInfo.numInnerThreads > 1 ? clodIterationMeshoptimizer : nullptr);

  // patch last lod level's counts
  geometry.lodLevelsCount = uint32_t(geometry.lodMesh.lodLevelGroupRanges.size());
  if(geometry.lodLevelsCount)
  {
    uint32_t groupCount =
        uint32_t(geometry.lodMesh.groupClusterRanges.size()) - geometry.lodMesh.lodLevelGroupRanges.back().offset;

    nvcluster_Range& lastLodLevelGroupRange = geometry.lodMesh.lodLevelGroupRanges.back();

    lastLodLevelGroupRange.count         = groupCount;
    geometry.lodLevels.back().groupCount = groupCount;

    if(lastLodLevelGroupRange.count != 1 || geometry.lodMesh.groupClusterRanges[lastLodLevelGroupRange.offset].count != 1)
    {
      assert(0);
      LOGE("clodBuild failed: last lod level has more than one cluster\n");
      std::exit(-1);
    }
  }

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.lodMesh.triangleVertices.shrink_to_fit();
  geometry.lodMesh.clusterTriangleRanges.shrink_to_fit();
  geometry.lodMesh.clusterGeneratingGroups.shrink_to_fit();
  geometry.lodMesh.clusterBoundingSpheres.shrink_to_fit();
  geometry.lodMesh.groupClusterRanges.shrink_to_fit();
  geometry.lodMesh.lodLevelGroupRanges.shrink_to_fit();
  geometry.groupLodLevels.shrink_to_fit();

  buildGeometryLodHierarchyMeshoptimizer(processingInfo, geometry);

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.lodHierarchy.nodes.shrink_to_fit();
  geometry.lodHierarchy.groupCumulativeBoundingSpheres.shrink_to_fit();
  geometry.lodHierarchy.groupCumulativeQuadricError.shrink_to_fit();
}

void Scene::buildGeometryLodHierarchyMeshoptimizer(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  // for each lod level build hierarchy

  uint32_t lodLevelCount = uint32_t(geometry.lodMesh.lodLevelGroupRanges.size());

  std::vector<nvcluster_Range> lodNodeRanges(lodLevelCount);

  // top root is first
  // lod-level many lod-roots next
  // then rest
  {
    uint32_t nodeOffset = 1 + lodLevelCount;

    for(uint32_t lodLevel = 0; lodLevel < lodLevelCount; lodLevel++)
    {
      const nvcluster_Range& lodGroupRange = geometry.lodMesh.lodLevelGroupRanges[lodLevel];

      // groups as leaves
      uint32_t nodeCount = lodGroupRange.count;

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
    geometry.lodHierarchy.nodes.resize(nodeOffset);
  }

  // build per-level trees
  nvutils::parallel_batches_pooled<1>(
      lodLevelCount,
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        uint32_t               lodLevel      = uint32_t(idx);
        const nvcluster_Range& lodGroupRange = geometry.lodMesh.lodLevelGroupRanges[lodLevel];
        const nvcluster_Range& lodNodeRange  = lodNodeRanges[lodLevel];

        // groups as leaves
        uint32_t nodeCount      = lodGroupRange.count;
        uint32_t nodeOffset     = lodNodeRange.offset;
        uint32_t lastNodeOffset = nodeOffset;

        for(uint32_t g = 0; g < nodeCount; g++)
        {
          uint32_t                    groupID = g + lodGroupRange.offset;
          nvclusterlod_HierarchyNode& node =
              nodeCount == 1 ? geometry.lodHierarchy.nodes[1 + lodLevel] : geometry.lodHierarchy.nodes[nodeOffset++];

          node                                   = {};
          node.clusterGroup.isClusterGroup       = 1;
          node.clusterGroup.clusterCountMinusOne = geometry.lodMesh.groupClusterRanges[groupID].count - 1;
          node.clusterGroup.group                = groupID;
          node.boundingSphere                    = geometry.lodHierarchy.groupCumulativeBoundingSpheres[groupID];
          node.maxClusterQuadricError            = geometry.lodHierarchy.groupCumulativeQuadricError[groupID];
        }
        // special case single node, directly stored to root section
        if(nodeCount == 1)
        {
          nodeOffset++;
        }

        // then nodes on top
        uint32_t depth          = 0;
        uint32_t iterationCount = nodeCount;

        std::vector<uint32_t>                   partitionedIndices;
        std::vector<nvclusterlod_HierarchyNode> oldNodes;

        while(iterationCount > 1)
        {
          uint32_t                    lastNodeCount = iterationCount;
          nvclusterlod_HierarchyNode* lastNodes     = &geometry.lodHierarchy.nodes[lastNodeOffset];

          // partition last nodes into children for new nodes
          partitionedIndices.resize(lastNodeCount);
          meshopt_spatialClusterPoints(partitionedIndices.data(), &lastNodes->boundingSphere.center.x, lastNodeCount,
                                       sizeof(nvclusterlod_HierarchyNode), m_config.preferredNodeWidth);

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
          nvclusterlod_HierarchyNode* newNodes = iterationCount == 1 ? &geometry.lodHierarchy.nodes[1 + lodLevel] :
                                                                       &geometry.lodHierarchy.nodes[nodeOffset];

          for(uint32_t n = 0; n < iterationCount; n++)
          {
            nvclusterlod_HierarchyNode& node          = newNodes[n];
            nvclusterlod_HierarchyNode* childrenNodes = &lastNodes[n * m_config.preferredNodeWidth];

            uint32_t childCount = std::min((n + 1) * m_config.preferredNodeWidth, lastNodeCount) - n * m_config.preferredNodeWidth;

            node                             = {};
            node.children.isClusterGroup     = 0;
            node.children.childCountMinusOne = childCount - 1;
            node.children.childOffset        = lastNodeOffset + n * m_config.preferredNodeWidth;
            node.maxClusterQuadricError      = 0;

            for(uint32_t c = 0; c < childCount; c++)
            {
              node.maxClusterQuadricError = std::max(node.maxClusterQuadricError, childrenNodes[c].maxClusterQuadricError);
            }

            meshopt_Bounds merged =
                meshopt_computeSphereBounds(&childrenNodes[0].boundingSphere.center.x, childCount, sizeof(nvclusterlod_HierarchyNode),
                                            &childrenNodes[0].boundingSphere.radius, sizeof(nvclusterlod_HierarchyNode));

            node.boundingSphere.center.x = merged.center[0];
            node.boundingSphere.center.y = merged.center[1];
            node.boundingSphere.center.z = merged.center[2];
            node.boundingSphere.radius   = merged.radius;
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
    meshopt_Bounds merged = meshopt_computeSphereBounds(&geometry.lodHierarchy.nodes[1].boundingSphere.center.x,
                                                        lodLevelCount, sizeof(nvclusterlod_HierarchyNode),
                                                        &geometry.lodHierarchy.nodes[1].boundingSphere.radius,
                                                        sizeof(nvclusterlod_HierarchyNode));

    nvclusterlod_HierarchyNode& node          = geometry.lodHierarchy.nodes[0];
    nvclusterlod_HierarchyNode* childrenNodes = &geometry.lodHierarchy.nodes[1];

    node                             = {};
    node.children.isClusterGroup     = 0;
    node.children.childCountMinusOne = lodLevelCount - 1;
    node.children.childOffset        = 1;
    node.boundingSphere.center.x     = merged.center[0];
    node.boundingSphere.center.y     = merged.center[1];
    node.boundingSphere.center.z     = merged.center[2];
    node.boundingSphere.radius       = merged.radius;
    node.maxClusterQuadricError      = 0;

    for(uint32_t c = 0; c < lodLevelCount; c++)
    {
      node.maxClusterQuadricError = std::max(node.maxClusterQuadricError, childrenNodes[c].maxClusterQuadricError);
    }
  }
}

void Scene::buildGeometryClusterLodNvLib(const ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  nvclusterlod_Result result;

  nvclusterlod_MeshInput lodMeshInput;
  lodMeshInput.decimationFactor = m_config.lodLevelDecimationFactor;
  lodMeshInput.triangleCount    = uint32_t(geometry.globalTriangles.size());
  lodMeshInput.triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(geometry.globalTriangles.data());
  lodMeshInput.vertexCount      = uint32_t(geometry.vertices.size());
  lodMeshInput.vertexStride     = sizeof(glm::vec4);
  lodMeshInput.vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(geometry.vertices.data());

  lodMeshInput.clusterConfig = getClusterConfig();
  lodMeshInput.groupConfig   = getGroupConfig();

  result = nvclusterlod::generateLodMesh(processingInfo.lodContext, lodMeshInput, geometry.lodMesh);
  if(result != NVCLUSTERLOD_SUCCESS)
  {
    assert(0);
    LOGE("nvclusterlod::generateLodMesh failed: %d\n", result);
    std::exit(-1);
  }

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.lodMesh.triangleVertices.shrink_to_fit();
  geometry.lodMesh.clusterTriangleRanges.shrink_to_fit();
  geometry.lodMesh.clusterGeneratingGroups.shrink_to_fit();
  geometry.lodMesh.clusterBoundingSpheres.shrink_to_fit();
  geometry.lodMesh.groupQuadricErrors.shrink_to_fit();
  geometry.lodMesh.groupClusterRanges.shrink_to_fit();
  geometry.lodMesh.lodLevelGroupRanges.shrink_to_fit();


  nvclusterlod_HierarchyInput hierarchyInput = {};
  hierarchyInput.clusterCount                = uint32_t(geometry.lodMesh.clusterBoundingSpheres.size());
  hierarchyInput.clusterBoundingSpheres      = geometry.lodMesh.clusterBoundingSpheres.data();
  hierarchyInput.clusterGeneratingGroups     = geometry.lodMesh.clusterGeneratingGroups.data();
  hierarchyInput.groupClusterRanges          = geometry.lodMesh.groupClusterRanges.data();
  hierarchyInput.groupQuadricErrors          = geometry.lodMesh.groupQuadricErrors.data();
  hierarchyInput.lodLevelGroupRanges         = geometry.lodMesh.lodLevelGroupRanges.data();
  hierarchyInput.lodLevelCount               = uint32_t(geometry.lodMesh.lodLevelGroupRanges.size());
  hierarchyInput.groupCount                  = int32_t(geometry.lodMesh.groupClusterRanges.size());

  // required to later traverse the lod hierarchy in parallel
  // Note we build hierarchies over each lod level's groups
  // and then join them together to a single hierarchy.
  // This is key to the parallel traversal algorithm that traverse multiple lod levels
  // at once.

  result = nvclusterlod::generateLodHierarchy(processingInfo.lodContext, hierarchyInput, geometry.lodHierarchy);
  if(result != NVCLUSTERLOD_SUCCESS)
  {
    assert(0);
    LOGE("nvclusterlod::generateLodHierarchy failed: %d\n", result);
    std::exit(-1);
  }

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.lodHierarchy.nodes.shrink_to_fit();
  geometry.lodHierarchy.groupCumulativeBoundingSpheres.shrink_to_fit();
  geometry.lodHierarchy.groupCumulativeQuadricError.shrink_to_fit();

  // no longer needed
  geometry.lodMesh.groupQuadricErrors = {};

  // setup lod level details
  shaderio::LodLevel initLevel{};
  initLevel.minBoundingSphereRadius = FLT_MAX;
  initLevel.minMaxQuadricError      = FLT_MAX;

  geometry.lodLevelsCount = uint32_t(geometry.lodMesh.lodLevelGroupRanges.size());
  geometry.lodLevels.resize(geometry.lodMesh.lodLevelGroupRanges.size(), initLevel);
  geometry.groupLodLevels.resize(geometry.lodMesh.groupClusterRanges.size());

  for(size_t level = 0; level < geometry.lodMesh.lodLevelGroupRanges.size(); level++)
  {
    nvcluster_Range groupRange = geometry.lodMesh.lodLevelGroupRanges[level];

    geometry.lodLevels[level].groupCount  = groupRange.count;
    geometry.lodLevels[level].groupOffset = groupRange.offset;

    for(size_t g = groupRange.offset; g < groupRange.offset + groupRange.count; g++)
    {
      nvcluster_Range clusterRange = geometry.lodMesh.groupClusterRanges[g];

      geometry.groupLodLevels[g] = uint8_t(level);

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
          std::min(geometry.lodLevels[level].minBoundingSphereRadius,
                   geometry.lodHierarchy.groupCumulativeBoundingSpheres[g].radius);
      geometry.lodLevels[level].minMaxQuadricError =
          std::min(geometry.lodLevels[level].minMaxQuadricError, geometry.lodHierarchy.groupCumulativeQuadricError[g]);
    }
  }
}

}  // namespace lodclusters