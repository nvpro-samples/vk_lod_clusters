/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  This compute shader initializes the traversal queue with the 
  root nodes of the lod hierarchy of rendered instances.

  A thread represents one instance.

  NOT compatible with USE_BLAS_SHARING, see `traversal_init_blas_reuse.comp.glsl`
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

#if TARGETS_RASTERIZATION && USE_TWO_PASS_CULLING
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

#if TARGETS_RASTERIZATION && USE_TWO_PASS_REJECT_LISTS
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

#if TARGETS_RASTERIZATION && USE_TWO_PASS_REJECT_LISTS
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
  
  uint blasBuildIndex = BLAS_BUILD_INDEX_LOWDETAIL;

#if USE_INSTANCE_OCCLUSION_CULLING
  bool useOcclusion = true;
#else
  bool useOcclusion = false;
#endif
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
#if TARGETS_RASTERIZATION && USE_TWO_PASS_CULLING
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

#if TARGETS_RASTERIZATION
  // Solo-instance filter: ~0u disables the filter.
  bool soloFiltered = view.visFilterInstanceID != ~0u && instanceID != view.visFilterInstanceID;
  if (soloFiltered)
  {
    isVisible = false;
  }
#else
  const bool soloFiltered = false;
#endif

#if TARGETS_RASTERIZATION && USE_TWO_PASS_REJECT_LISTS
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
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    && isVisible
  #endif
    ;
    
  bool traverseRootNode = isRenderable;

  if (isRenderable)
  {
    // We test if we are only using the furthest lod.
    // If that is true, then we can skip lod traversal completely and
    // straight enqueue the lowest detail cluster directly.    
    
    uint rootNodePacked = geometry.nodes.d[0].packed;
    
    uint childOffset        = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);
    uint childCountMinusOne = PACKED_GET(rootNodePacked, Node_packed_nodeChildCountMinusOne);
    
    // test if the second to last lod needs to be traversed
    uint childNodeIndex     = (childCountMinusOne > 1 ? (childCountMinusOne - 1) : 0);
    Node childNode          = geometry.nodes.d[childOffset + childNodeIndex];
    TraversalMetric traversalMetric = childNode.traversalMetric;
  
    mat4x3 worldMatrix = transpose(instance.worldMatrix);
    float uniformScale = computeUniformScale(worldMatrix);
    float errorScale   = 1.0;
  #if USE_CULLING && TARGETS_RAY_TRACING
    if (visibilityState == 0) errorScale = build.culledErrorScale;
  #endif
  
    mat4 transform = build.traversalViewMatrix * toMat4(worldMatrix);
  
    // if there is no need to traverse the pen ultimate lod level,
    // then just insert the last lod level node's cluster directly.
    // A geometry with a single lod level has no pen ultimate level, its only level
    // is the lowest detail one. `instance_classify_lod.comp.glsl` classifies those
    // the same way, so both traversal init variants agree.
    if (childCountMinusOne == 0 || !testForTraversal(mat4x3(transform), uniformScale, traversalMetric, errorScale))
    {
    
    #if TARGETS_RAY_TRACING
      // we don't need to add a cluster because we always add it
      // implictly through the use of the low detail BLAS.
      
    #elif TARGETS_RASTERIZATION
      // lowest detail lod is guaranteed to have only one cluster.
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
    #endif

      // we can skip adding the node for traversal
      traverseRootNode = false;
    }
  }

  uvec4 voteNodes = subgroupBallot(traverseRootNode);  
  
  uint offsetNodes = 0;
  if (subgroupElect())
  {
    offsetNodes = atomicAdd(buildRW.traversalNodeWriteCounter, subgroupBallotBitCount(voteNodes));
  }
  
  offsetNodes = subgroupBroadcastFirst(offsetNodes);  
  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);
      
  if (traverseRootNode && offsetNodes < build.maxTraversalInfos)
  {
    uint rootNodePacked = geometry.nodes.d[0].packed;

    TraversalInfo traversalInfo;
    traversalInfo.instanceID = instanceID;
    traversalInfo.packedNode = rootNodePacked;

    build.traversalNodeInfos.d[offsetNodes] = packTraversalInfo(traversalInfo);
  }

#if TARGETS_RAY_TRACING
  if (isValid) {
    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);  
    build.instanceBuildInfos.d[instanceID].clusterReferencesCount = 0;
    build.instanceBuildInfos.d[instanceID].blasBuildIndex         = blasBuildIndex;
    
    // We might want to remove the instance completely if not visible, or just use the low detail blas
  #if USE_CULLING && USE_FORCED_INVISIBLE_CULLING && FORCE_INVISIBLE_CULLED_REMOVES_INSTANCE
    if(!isVisible && build.frameIndex != 0){
      // first frame must always have a valid BLAS due to TLAS BUILD, other frames are TLAS UPDATE
      build.tlasInstances.d[instanceID].blasReference             = 0;
    }
    else
  #endif
    {
      build.tlasInstances.d[instanceID].blasReference             = geometry.lowDetailBlasAddress;
    }
  }
#elif TARGETS_RASTERIZATION
  // drives the two pass culling and VISUALIZE_DISCRETE_LOD
  if (build.cullPass == 0 && isValid) {
    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);
  }
#endif
}