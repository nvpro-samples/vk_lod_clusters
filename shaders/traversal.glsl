/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Splits the discrete lod tag from a traversal item's instanceID.
// Returns true if the item was seeded at a discrete lod level.
bool unpackTraversalDiscrete(inout uint instanceID)
{
#if USE_DISCRETE_LOD
  bool isDiscrete = (instanceID & TRAVERSAL_DISCRETE_BIT) != 0;
  instanceID      = instanceID & ~TRAVERSAL_DISCRETE_BIT;
  return isDiscrete;
#else
  return false;
#endif
}

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

// The lod levels an instance may use. `lodLevelMin` is the highest potential
// detail, `lodLevelMax` the lowest.
struct InstanceLodClassification
{
  uint lodLevelMin;
  uint lodLevelMax;
  // false if no lod level was coarse enough, the instance is then far enough
  // away that only the lowest detail is required. `lodLevelMin` stays 0.
  bool lodLevelMinFound;
};

// Determines the lod level range of an instance from the geometry's root node,
// which has one child node per lod level.
//
// `startLevel` skips levels the caller is not interested in, `lodLevelMin` then
// stays 0 if no scanned level is coarse enough.
// `findMax` also computes `lodLevelMax`, which needs the per lod level minimum
// sphere data. Without it `lodLevelMax` is the geometry's last lod level.
InstanceLodClassification classifyInstanceLod(Geometry geometry,
                                              mat4x3   instanceToEye,
                                              vec3     oViewPos,
                                              float    uniformScale,
                                              float    errorScale,
                                              uint     startLevel,
                                              bool     findMax)
{
  uint rootNodePacked = geometry.nodes.d[0].packed;
  uint childOffset    = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);

  InstanceLodClassification classification;
  classification.lodLevelMin      = 0;
  classification.lodLevelMax      = geometry.lodLevelsCount - 1;
  classification.lodLevelMinFound = false;

  bool findMin = true;

  for (uint lodLevel = startLevel; lodLevel < geometry.lodLevelsCount; lodLevel++)
  {
    Node childNode                  = geometry.nodes.d[childOffset + lodLevel];
    TraversalMetric traversalMetric = childNode.traversalMetric;

    // During lod traversal, we use the offline accumulated maximum sphere of the cluster groups stored into lod nodes
    // to test whether there is potentially something to be rendered. We want to optimize for the highest
    // error we get away with. So we test in detail if something is "coarse enough" (error > threshold).
    // `childNode.traversalMetric` provides the data for the maximum sphere.
    //
    // An actual cluster is rendered if
    //  1) cluster's group            ` error over distance > threshold` (group's lod level is coarse enough)
    //  2) clusters' generating group `!error over distance > threshold` (generating group is in lod level - 1)

    // Example:
    //
    //  lod level                   | 0 | 1 | 2 | 3 | 4
    //  testForTraversal(maxSphere) | - | x | x | x | x
    //
    // In the example it's guaranteed that at least 1 cluster could be rendered at lod level 1.
    //   1) is ensured due to one group being represented within the accumulated maximum sphere
    //   2) is ensured because no such parent group can exist, otherwise `testForTraversal(maxSphere)`
    //      would have evaluated to true for lod level 0.
    //
    // The first transition of the metric determines the lod level of a certain surface region.
    // This serves as the "fine enough".
    // Clusters from higher, less detailed, lod levels (e.g 2,3,4) of the same
    // region will not trigger, because their generating group's will not pass 2)
    //
    // We will use this reasoning to find the highest possible lod level as well.

    if (findMin && testForTraversal(instanceToEye, uniformScale, traversalMetric, errorScale))
    {
      findMin = false;
      classification.lodLevelMin      = lodLevel;
      classification.lodLevelMinFound = true;

      if (!findMax) break;
    }

    if (findMax && !findMin)
    {
      // This time we use the smallest possible sphere for each lod level.
      //
      // The smallest possible sphere was pre-computed for the geometry for each lod level.
      // We took the smallest radius, and the smallest `maxQuadraticError` found in any group,
      // and the sphere is put at the furthest possible distance from the camera,
      // while still within the maximum sphere.
      // These conditions ensure that nothing with a smaller `error over distance`
      // behavior can exist.

      vec3 oSpherePos = TraversalMetric_getSphere(traversalMetric);
      vec3 oViewDir   = normalize(oSpherePos - oViewPos);

      oSpherePos.xyz += oViewDir * (traversalMetric.boundingSphereRadius - geometry.lodLevels.d[lodLevel].minBoundingSphereRadius);

      traversalMetric.boundingSphereX = oSpherePos.x;
      traversalMetric.boundingSphereY = oSpherePos.y;
      traversalMetric.boundingSphereZ = oSpherePos.z;
      traversalMetric.boundingSphereRadius = geometry.lodLevels.d[lodLevel].minBoundingSphereRadius;
      traversalMetric.maxQuadricError      = geometry.lodLevels.d[lodLevel].minMaxQuadricError;

      // Example:
      //
      //  lod level                   | 0 | 1 | 2 | 3 | 4
      //  testForTraversal(minSphere) | - | - | - | x | x
      //
      // If even the smallest possible group in lod level 3 is coarse enough, it
      // means there cannot be a group that would first transition in lod level 4
      //
      // Therefore lod level 3 is guaranteed to be the last active lod level.

      if (testForTraversal(instanceToEye, uniformScale, traversalMetric, errorScale))
      {
        classification.lodLevelMax = lodLevel;
        break;
      }
    }
  }

  return classification;
}

#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)

// Node visibility test used by the lod traversal. Unlike clusters, nodes are
// tested against the best available hiz.
bool queryWasVisible(mat4x3 instanceTransform, vec3 bboxMin, vec3 bboxMax)
{
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;

#if USE_NODE_OCCLUSION_CULLING
  bool useOcclusion = true;
#else
  bool useOcclusion = false;
#endif

#if USE_TWO_PASS_CULLING

  // clusters are always first tested against last hiz
  // node's should be tested against best available hiz
  bool useLast =  build.cullPass == 0;

  bool inFrustum = intersectFrustum(useLast ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum &&
    (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, useLast ? 0 : 1)));
#else
  // always test against last frame visiblity
  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum &&
    (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, 0)));
#endif

  return isVisible;
}

// Coarse group test for the flat discrete lod composition, which has no node
// hierarchy above the groups to cull against. The group's lod metric sphere is
// used as bounds, it is looser than the node bboxes the traversal would use,
// but comes for free with the group header.
bool queryGroupWasVisible(mat4x3 instanceTransform, TraversalMetric metric)
{
  vec3 sphereCenter = TraversalMetric_getSphere(metric);
  vec3 sphereExtent = vec3(metric.boundingSphereRadius);

  return queryWasVisible(instanceTransform, sphereCenter - sphereExtent, sphereCenter + sphereExtent);
}

// Cluster visibility test used by the lod traversal and the flat discrete lod
// composition. Also classifies the cluster for sw rasterization.
bool queryClusterWasVisible(mat4x3 instanceTransform, BBox bbox, inout bool outRenderClusterSW)
{
  vec3 bboxMin = bbox.lo;
  vec3 bboxMax = bbox.hi;

  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;

#if USE_CLUSTER_OCCLUSION_CULLING
  bool useOcclusion = true;
#else
  bool useOcclusion = false;
#endif

#if USE_TWO_PASS_REJECT_LISTS
  // A group only reaches this kernel in the pass that first traverses it, so a
  // single test against that pass' hiz is enough. The clusters the first pass
  // rejected are re-tested by `traversal_reject_clusters.comp.glsl` instead.
  bool useLast   = build.cullPass == 0;
  bool inFrustum = intersectFrustum(useLast ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum &&
    (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, useLast ? 0 : 1)));
#else
  // test if visible in last frame
  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum &&
    (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, 0)));

#if USE_TWO_PASS_CULLING
  if (build.cullPass == 1)
  {
    // in second pass also test against current visibility

    if (isVisible) {
      // was rendered in first pass already
      isVisible = false;
    }
    else {
      // test against current
      inFrustum = intersectFrustum(build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
      isVisible = inFrustum &&
        (!clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, 1)));
    }
  }
#endif
#endif

#if USE_SW_RASTER
  // check if sw rasterization is okay to use (not near/far clipped and smaller than threshold)

  // TODO should embed this relative longest edge in bbox instead
  vec3 bboxDim       = bboxMax - bboxMin;
  float relativeSize = bbox.longestEdge / length(bboxDim);

  if (isVisible && clipMin.z > 0 && clipMax.z < 1 && clipValid && !intersectSize(clipMin, clipMax, build.swRasterThreshold, relativeSize))
  {
    outRenderClusterSW = true;
  }
#endif

  return isVisible;
}

#endif

#if TARGETS_RASTERIZATION

// Whether a cluster requires alpha testing, based on instance, group and cluster state.
bool queryClusterUsesAlpha(uint instanceID, Group_in groupRef, Group group, uint clusterIndex)
{
#if HAS_ALPHA_TEST
  if (instances[instanceID].opaqueStatus == SHADERIO_OPAQUE_STATUS_ALPHAMASKED)
  {
    return true;
  }
  if (instances[instanceID].opaqueStatus == SHADERIO_OPAQUE_STATUS_MIXED)
  {
    // check group state bit first if all clusters are alphamasked
    uint groupState       = group.stateBits;
    bool alphaMasked      = (groupState & CLUSTER_STATE_ALPHAMASKED) != 0;
    bool alphaMaskedMixed = (groupState & CLUSTER_STATE_ALPHAMASKED_MIXED) != 0;

    if (alphaMasked && !alphaMaskedMixed)
    {
      return true;
    }
    else if (alphaMasked && alphaMaskedMixed)
    {
      // check cluster state bits if not all clusters are alphamasked
      return (Group_getClusterState(groupRef, clusterIndex) & CLUSTER_STATE_ALPHAMASKED) != 0;
    }
  }
#endif
  return false;
}

#endif

// Whether a geometry takes part in the sharing election, which needs at least two
// instances. Caching does not depend on this, it only needs `lodLevelMin`.
bool testForBlasSharing(Geometry geometry)
{
#if !USE_BLAS_SHARING
  return false;
#elif USE_BLAS_CACHING
  return geometry.instancesCount >= 1;
#else
  return geometry.instancesCount >= 2;
#endif
}
