/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  USE_DISCRETE_LOD variant of `traversal_init.comp.glsl` for rasterization.

  Next to initializing the traversal queue with the root nodes of the lod
  hierarchy, it classifies the lod levels an instance uses. Instances that get
  away with a single, fully resident, discrete lod level seed the traversal at
  that level's node instead of the root, tagged with TRAVERSAL_DISCRETE_BIT so
  the lod metric is skipped below it.

  A thread represents one instance.

  Rasterization only, it is the spiritual equivalent to ray tracing's
  BLAS_CACHING
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_RENDERMATERIALS_SSBO, set = 0) buffer renderMaterialsBuffer
{
  RenderMaterial materials[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

#if USE_TWO_PASS_CULLING
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];
#else
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
#endif

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};


////////////////////////////////////////////

layout(local_size_x=TRAVERSAL_INIT_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);

#if USE_TWO_PASS_REJECT_LISTS
  // the second pass only revisits the instances that the first pass rejected
  bool useRejects   = build.cullPass == 1;
  uint numInstances = useRejects ? min(build.rejectInstanceCounter, build.numRenderInstances) : build.numRenderInstances;
#else
  const bool useRejects = false;
  uint numInstances     = build.numRenderInstances;
#endif

  bool isValid      = threadID < numInstances;
  uint instanceLoad = isValid ? threadID : 0;
  uint instanceID   = instanceLoad;

#if USE_TWO_PASS_REJECT_LISTS
  if (useRejects)
  {
    instanceLoad = build.rejectInstances.d[instanceLoad];
    instanceID   = instanceLoad;
  }
  else
#endif
  {
#if USE_SORTING
    instanceLoad = build.instanceSortValues.d[instanceLoad];
    instanceID   = instanceLoad;
#endif
  }

  RenderInstance instance = instances[instanceLoad];
  uint geometryID = instance.geometryID;
  Geometry geometry = geometries[geometryID];
  
#if USE_INSTANCE_OCCLUSION_CULLING
  bool useOcclusion = true;
#else
  bool useOcclusion = false;
#endif
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
#if USE_TWO_PASS_CULLING
  bool inFrustum = intersectFrustum( build.cullPass == 0 ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, geometry.bbox.lo, geometry.bbox.hi, transpose(instance.worldMatrix), clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, build.cullPass)));

#if !USE_TWO_PASS_REJECT_LISTS
  // if smallish and was already drawn, don't process again
  if (build.cullPass == 1 && isVisible && clipValid && !intersectSize(clipMin, clipMax, CULL_SECOND_PASS_MIN_PIXEL_SIZE) && ((uint(build.instanceVisibility.d[instanceLoad]) & INSTANCE_VISIBLE_BIT) != 0)) {
    isVisible = false;
  }
#endif

#else
  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, geometry.bbox.lo, geometry.bbox.hi, transpose(instance.worldMatrix), clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, 0)));
#endif

  // Solo-instance filter: ~0u disables the filter.
  bool soloFiltered = view.visFilterInstanceID != ~0u && instanceID != view.visFilterInstanceID;
  if (soloFiltered)
  {
    isVisible = false;
  }

#if USE_TWO_PASS_REJECT_LISTS
  {
    // Record instances the first pass could not draw. Only those within the current
    // frustum can reappear once the hiz is updated, everything else would fail the
    // second pass anyway, so filter here and keep the list short.
    bool reject = isValid && build.cullPass == 0 && !isVisible && !soloFiltered
                  && intersectFrustumOnly(build.cullViewProjMatrix, geometry.bbox.lo, geometry.bbox.hi, transpose(instance.worldMatrix));

    uvec4 voteReject   = subgroupBallot(reject);
    uint  offsetReject = 0;
    if (subgroupElect())
    {
      offsetReject = atomicAdd(buildRW.rejectInstanceCounter, subgroupBallotBitCount(voteReject));
    }
    offsetReject = subgroupBroadcastFirst(offsetReject) + subgroupBallotExclusiveBitCount(voteReject);

    if (reject && offsetReject < build.numRenderInstances)
    {
      build.rejectInstances.d[offsetReject] = instanceID;
    }
  }
#endif

  uint visibilityState = isVisible ? INSTANCE_VISIBLE_BIT : 0;
  
  bool isRenderable = isValid
  #if USE_CULLING
    && isVisible
  #endif
    ;
    
  bool traverseRootNode   = isRenderable;
  bool useDiscreteLod     = false;
  uint discreteLodLevel   = 0;
  // the geometry's root node has one child node per lod level, the discrete lod
  // seeds the traversal with the child of the level it uses
  uint discreteNodePacked = 0;

  if (isRenderable)
  {
    mat4x3 worldMatrix  = transpose(instance.worldMatrix);
    mat4x3 worldMatrixI = transpose(instance.worldMatrixI);
    float uniformScale  = computeUniformScale(worldMatrix);
    float errorScale    = 1.0;

    mat4 transform = build.traversalViewMatrix * toMat4(worldMatrix);
    vec3 oViewPos  = (worldMatrixI * vec4(view.viewPos.xyz, 1));

    uint geometryLodLevelMax = geometry.lodLevelsCount - 1;

    // Only the last lod levels are allowed to use the flat composition, start
    // the classification one level ahead of those, so we can tell whether the
    // instance needs more detail than the range covers.
    uint tailLevel  = geometryLodLevelMax - min(build.discreteEnabledLevels, geometryLodLevelMax);
    uint startLevel = tailLevel > 0 ? tailLevel - 1 : 0;

    // `lodLevelMax` is only needed when the range can actually reject something.
    // A range that spans the whole geometry never does, so skip the extra
    // per-level minimum sphere test in the classification.
    bool findMax = build.discreteLodRange <= geometryLodLevelMax;

    InstanceLodClassification classification =
        classifyInstanceLod(geometry, mat4x3(transform), oViewPos, uniformScale, errorScale, startLevel, findMax);

    uint lodLevelMin = classification.lodLevelMin;

    // no level was coarse enough, so the instance is beyond the last level's
    // error and only needs the lowest detail
    if (!classification.lodLevelMinFound || lodLevelMin == geometryLodLevelMax)
    {
      // The lowest detail lod level is guaranteed to have only one cluster and
      // is always resident, insert it directly.
      // A geometry with a single lod level has no other level to use.
      bool useAlpha = false;
      bool useSW = false;

    #if HAS_ALPHA_TEST
      useAlpha = (uint(instance.lowDetailClusterStateBits) & CLUSTER_STATE_ALPHAMASKED) != 0;
    #endif

    #if USE_SW_RASTER
      float relativeSize = geometry.bbox.longestEdge;
      if (isVisible && clipValid && clipMin.z > 0 && clipMax.z < 1 && !intersectSize(clipMin, clipMax, build.swRasterThreshold, relativeSize))
      {
        useSW = true;
      }
    #endif

      rasterBinning(geometry.lowDetailClusterID, instanceID, useAlpha, useSW, true);

      visibilityState |= INSTANCE_USES_LOWDETAIL_BIT;
      traverseRootNode = false;
    }
    else if (lodLevelMin >= tailLevel && lodLevelMin >= uint(geometry.discreteLodLevel))
    {
      // The instance gets away with a single discrete lod level that is fully
      // resident. Rendering all clusters of that level yields at least the
      // detail the lod traversal would have produced.
      // `classifyInstanceLod` leaves `lodLevelMax` at the geometry's last level
      // when it did not look for it, so a range that spans the whole geometry
      // passes here without needing a special case.
      useDiscreteLod = (classification.lodLevelMax - lodLevelMin) < build.discreteLodRange;

      discreteLodLevel = lodLevelMin;
      traverseRootNode = !useDiscreteLod;

      if (useDiscreteLod)
      {
        visibilityState |= INSTANCE_USES_DISCRETE_BIT;

        uint rootNodePacked = geometry.nodes.d[0].packed;
        uint childOffset    = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);
        discreteNodePacked  = geometry.nodes.d[childOffset + discreteLodLevel].packed;

        // The instance only touches the groups of its own lod level, so the
        // coarser levels would age out and break the geometry's residency level.
        // The streaming age filter keeps everything from here on alive.
        atomicMin(build.geometryCachedInfos.d[geometryID].cachedLevel, discreteLodLevel);
      }
    }
  }

#if USE_RENDER_STATS
  {
    uint countDiscrete = subgroupBallotBitCount(subgroupBallot(useDiscreteLod));
    if (subgroupElect() && countDiscrete != 0)
    {
      atomicAdd(readback.numDiscreteInstances, countDiscrete);
    }
  }
#endif

  // Seed the traversal with the lod level's node. The lod metric is skipped
  // within that subtree (TRAVERSAL_DISCRETE_BIT), but we keep the hierarchical
  // culling of the nodes.
  bool seedGroup = useDiscreteLod && PACKED_GET(discreteNodePacked, Node_packed_isGroup) != 0;
  bool seedNode  = useDiscreteLod && !seedGroup;

  bool enqueueNode = traverseRootNode || seedNode;

  uvec4 voteNodes = subgroupBallot(enqueueNode);  
  
  uint offsetNodes = 0;
  if (subgroupElect())
  {
    offsetNodes = atomicAdd(buildRW.traversalNodeWriteCounter, subgroupBallotBitCount(voteNodes));
  }
  
  offsetNodes = subgroupBroadcastFirst(offsetNodes);  
  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);
      
  if (enqueueNode && offsetNodes < build.maxTraversalInfos)
  {
    TraversalInfo traversalInfo;
    traversalInfo.instanceID = seedNode ? (instanceID | TRAVERSAL_DISCRETE_BIT) : instanceID;
    traversalInfo.packedNode = seedNode ? discreteNodePacked : geometry.nodes.d[0].packed;

    build.traversalNodeInfos.d[offsetNodes] = packTraversalInfo(traversalInfo);
  }

  // a lod level whose root is a leaf goes straight into the group queue
  if (subgroupAny(seedGroup))
  {
    uvec4 voteGroups = subgroupBallot(seedGroup);

    uint offsetGroups = 0;
    if (subgroupElect())
    {
      offsetGroups = atomicAdd(buildRW.traversalGroupWriteCounter, subgroupBallotBitCount(voteGroups));
    }

    offsetGroups = subgroupBroadcastFirst(offsetGroups);
    offsetGroups += subgroupBallotExclusiveBitCount(voteGroups);

    if (seedGroup && offsetGroups < build.maxTraversalInfos)
    {
      TraversalInfo traversalInfo;
      traversalInfo.instanceID = instanceID | TRAVERSAL_DISCRETE_BIT;
      traversalInfo.packedNode = discreteNodePacked;

      build.traversalGroupInfos.d[offsetGroups] = packTraversalInfo(traversalInfo);
    }
  }

  // `instanceVisibility` also drives the two pass culling and VISUALIZE_DISCRETE_LOD
  if (isValid && build.cullPass == 0) {
    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);
  }
}