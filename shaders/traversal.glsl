/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#define FLT_MAX 3.402823466e+38f

TraversalInfo unpackTraversalInfo(uint64_t packed64)
{
  u32vec2       data = unpack32(packed64);
  TraversalInfo info;
  info.instanceID = data.x;
  info.packedNode = data.y;
  return info;
}
uint64_t packTraversalInfo(TraversalInfo info)
{
  return pack64(u32vec2(info.instanceID, info.packedNode));
}

#if TARGETS_RASTERIZATION
// Subgroup-packed enqueue into renderClusterInfos / alpha / SW lists (see traversal_run_groups).
void rasterBinning(uint clusterID, uint instanceID, bool useAlpha, bool useSW, bool renderClusterAny)
{
  bool renderCluster = false;
  bool renderClusterSW = false;
  bool renderClusterAlpha = false;
  bool renderClusterAlphaSW = false;

#if USE_SW_RASTER || HAS_ALPHA_TEST
  if (renderClusterAny)
  {
#if USE_SW_RASTER
    if (useSW)
    {
#if HAS_ALPHA_TEST
      if (useAlpha)
      {
        renderClusterAlphaSW = true;
      }
      else
#endif
      {
        renderClusterSW = true;
      }
    }
    else
#endif
    {
#if HAS_ALPHA_TEST
      if (useAlpha)
      {
        renderClusterAlpha = true;
      }
      else
#endif
      {
        renderCluster = true;
      }
    }
  }
#else
  renderCluster = renderClusterAny;
#endif

#if USE_SW_RASTER
  uvec4 voteClustersSW = subgroupBallot(renderClusterSW);
  uint countClustersSW = subgroupBallotBitCount(voteClustersSW);
#if HAS_ALPHA_TEST
  uvec4 voteClustersAlphaSW = subgroupBallot(renderClusterAlphaSW);
  uint countClustersAlphaSW = subgroupBallotBitCount(voteClustersAlphaSW);
#endif
#endif

  uvec4 voteClusters = subgroupBallot(renderCluster);
  uint countClusters = subgroupBallotBitCount(voteClusters);
#if HAS_ALPHA_TEST
  uvec4 voteClustersAlpha = subgroupBallot(renderClusterAlpha);
  uint countClustersAlpha = subgroupBallotBitCount(voteClustersAlpha);
#endif

  uint offsetClusters = 0;
  uint offsetClustersSW = 0;
  uint offsetClustersAlpha = 0;
  uint offsetClustersAlphaSW = 0;

  if (subgroupElect())
  {
    offsetClusters = atomicAdd(buildRW.renderClusterCounter, countClusters);
#if HAS_ALPHA_TEST
    offsetClustersAlpha = atomicAdd(buildRW.renderClusterCounterAlpha, countClustersAlpha);
#endif
#if USE_SW_RASTER
    offsetClustersSW = atomicAdd(buildRW.renderClusterCounterSW, countClustersSW);
#if HAS_ALPHA_TEST
    offsetClustersAlphaSW = atomicAdd(buildRW.renderClusterCounterAlphaSW, countClustersAlphaSW);
#endif
#endif
  }

  offsetClusters = subgroupBroadcastFirst(offsetClusters);
  offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);
  renderCluster = renderCluster && offsetClusters < build.maxRenderClusters;

#if HAS_ALPHA_TEST
  offsetClustersAlpha = subgroupBroadcastFirst(offsetClustersAlpha);
  offsetClustersAlpha += subgroupBallotExclusiveBitCount(voteClustersAlpha);
  renderClusterAlpha = renderClusterAlpha && offsetClustersAlpha < build.maxRenderClusters;
#endif

#if USE_SW_RASTER
  offsetClustersSW = subgroupBroadcastFirst(offsetClustersSW);
  offsetClustersSW += subgroupBallotExclusiveBitCount(voteClustersSW);
  renderClusterSW = renderClusterSW && offsetClustersSW < build.maxRenderClusters;
#if HAS_ALPHA_TEST
  offsetClustersAlphaSW = subgroupBroadcastFirst(offsetClustersAlphaSW);
  offsetClustersAlphaSW += subgroupBallotExclusiveBitCount(voteClustersAlphaSW);
  renderClusterAlphaSW = renderClusterAlphaSW && offsetClustersAlphaSW < build.maxRenderClusters;
#endif
#endif

  if (renderCluster
#if HAS_ALPHA_TEST
      || renderClusterAlpha
#endif
#if USE_SW_RASTER
      || renderClusterSW
#if HAS_ALPHA_TEST
      || renderClusterAlphaSW
#endif
#endif
     )
  {
    TraversalInfo info;
    info.instanceID = instanceID;
    info.packedNode = clusterID;
#if USE_SW_RASTER || HAS_ALPHA_TEST
    uint writeIndex;
    uint64_t writePointer;
#if USE_SW_RASTER
    if (useSW)
    {
#if HAS_ALPHA_TEST
      if (useAlpha)
      {
        writeIndex = offsetClustersAlphaSW;
        writePointer = uint64_t(build.renderClusterInfosAlphaSW);
      }
      else
#endif
      {
        writeIndex = offsetClustersSW;
        writePointer = uint64_t(build.renderClusterInfosSW);
      }
    }
    else
#endif
    {
#if HAS_ALPHA_TEST
      if (useAlpha)
      {
        writeIndex = offsetClustersAlpha;
        writePointer = uint64_t(build.renderClusterInfosAlpha);
      }
      else
#endif
      {
        writeIndex = offsetClusters;
        writePointer = uint64_t(build.renderClusterInfos);
      }
    }
    uint64s_inout(writePointer).d[writeIndex] = packTraversalInfo(info);
#else
    uint writeIndex = offsetClusters;
    uint64s_inout(build.renderClusterInfos).d[writeIndex] = packTraversalInfo(info);
#endif
  }
}
#endif

float computeUniformScale(mat4 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}

float computeUniformScale(mat4x3 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}

vec3 TraversalMetric_getSphere(TraversalMetric metric)
{
  return vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
}
void TraversalMetric_setSphere(inout TraversalMetric metric, vec3 sphere)
{
  metric.boundingSphereX = sphere.x;
  metric.boundingSphereY = sphere.y;
  metric.boundingSphereZ = sphere.z;
}

// key function for the lod metric evaluation
// returns true if error is over threshold ("coarse enough")
bool testForTraversal(mat4x3 instanceToEye, float uniformScale, TraversalMetric metric, float errorScale)
{
  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(vec3(instanceToEye * vec4(boundingSpherePos, 1.0f)));
  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;
  
  // error is over threshold, we are coarse enough
  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}

// variant of the above, assumes world space for view position and metric sphere position
bool testForTraversal(vec3 wViewPos, float uniformScale, TraversalMetric metric, float errorScale)
{
  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(wViewPos - boundingSpherePos);
  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;
  
  // error is over threshold, we are coarse enough
  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}

bool testForBlasSharing(Geometry geometry)
{
#if USE_BLAS_CACHING
  return geometry.instancesCount >= 1;
#else
  return geometry.instancesCount >= 2;
#endif
}
