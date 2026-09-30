/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Second cull pass companion to `traversal_run_groups.comp.glsl`, only used
  with USE_TWO_PASS_REJECT_LISTS.

  The first pass recorded, for every cluster group it traversed, which of its
  clusters the lod metric wanted but that failed the occlusion test
  (`build.rejectClusterMasks`, one bit per cluster, indexed by the group's slot
  in the first pass' `build.pass0GroupInfos`).

  The lod decision is identical in both passes, only the visibility answer
  changes. So this kernel does not redo any traversal or metric evaluation, it
  just re-tests the recorded clusters against the updated hiz and appends the
  survivors for rendering.

  Clusters that were drawn in the first pass have no bit set and cannot be drawn
  twice, which is what makes the first pass' hiz re-test unnecessary.

  one thread represents one cluster group of the first pass.
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

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;
};

#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};
#endif

////////////////////////////////////////////

layout(local_size_x=TRAVERSAL_REJECT_CLUSTERS_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////

// Unlike `queryWasVisible` this always tests the current frame, the reject bits
// already carry the first pass' answer.
bool queryIsVisible(mat4x3 instanceTransform, BBox bbox, inout bool outRenderClusterSW)
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

  bool inFrustum = intersectFrustum(build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum &&
    (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, 1)));

#if USE_SW_RASTER
  // check if sw rasterization is okay to use (not near/far clipped and smaller than threshold)
  vec3 bboxDim       = bboxMax - bboxMin;
  float relativeSize = bbox.longestEdge / length(bboxDim);

  if (isVisible && clipMin.z > 0 && clipMax.z < 1 && clipValid && !intersectSize(clipMin, clipMax, build.swRasterThreshold, relativeSize))
  {
    outRenderClusterSW = true;
  }
#endif

  return isVisible;
}

////////////////////////////////////////////

void main()
{
  uint threadReadIndex = getGlobalInvocationIndex(gl_GlobalInvocationID);
  if (threadReadIndex >= build.pass0GroupCount) return;

  uint rejectMask[REJECT_CLUSTER_MASK_WORDS];
  bool anyReject = false;
  [[unroll]] for (uint w = 0; w < REJECT_CLUSTER_MASK_WORDS; w++)
  {
    rejectMask[w] = build.rejectClusterMasks.d[threadReadIndex * REJECT_CLUSTER_MASK_WORDS + w];
    anyReject     = anyReject || rejectMask[w] != 0;
  }

  // most groups are either fully drawn or fully culled in the first pass
  if (!anyReject) return;

  TraversalInfo traversalInfo = unpackTraversalInfo(build.pass0GroupInfos.d[threadReadIndex]);
  uint instanceID             = traversalInfo.instanceID;
  // a discrete lod seed only forces the lod decision, which the reject bits already carry,
  // but the tag has to come off the instance id before it is used as an index
  unpackTraversalDiscrete(instanceID);
  uint groupIndex             = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
  uint groupClusterCount      = PACKED_GET(traversalInfo.packedNode, Node_packed_groupClusterCountMinusOne) + 1;

  uint geometryID   = instances[instanceID].geometryID;
  Geometry geometry = geometries[geometryID];

  mat4x3 worldMatrix = transpose(instances[instanceID].worldMatrix);

#if USE_STREAMING
  // the group was resident when the first pass traversed it, and residency
  // cannot change within a frame
  Group_in groupRef = Group_in(geometry.streamingGroupAddresses.d[groupIndex]);
#else
  Group_in groupRef = Group_in(geometry.preloadedGroups.d[groupIndex]);
#endif
  Group group = groupRef.d;

  for (uint clusterIndex = 0; clusterIndex < groupClusterCount; clusterIndex++)
  {
    bool rejected = (rejectMask[clusterIndex >> 5] & (1u << (clusterIndex & 31))) != 0;

    bool useAlpha         = false;
    bool useSW            = false;
    bool renderClusterAny = false;

    if (rejected)
    {
      BBox bbox        = Group_getClusterBBox(groupRef, clusterIndex);
      renderClusterAny = queryIsVisible(worldMatrix, bbox, useSW);

    #if HAS_ALPHA_TEST
      useAlpha = instances[instanceID].opaqueStatus == SHADERIO_OPAQUE_STATUS_ALPHAMASKED;
      if (instances[instanceID].opaqueStatus == SHADERIO_OPAQUE_STATUS_MIXED)
      {
        uint groupState       = group.stateBits;
        bool alphaMasked      = (groupState & CLUSTER_STATE_ALPHAMASKED) != 0;
        bool alphaMaskedMixed = (groupState & CLUSTER_STATE_ALPHAMASKED_MIXED) != 0;

        if (alphaMasked && !alphaMaskedMixed)
        {
          useAlpha = true;
        }
        else if (alphaMasked && alphaMaskedMixed)
        {
          uint clusterState = Group_getClusterState(groupRef, clusterIndex);
          if ((clusterState & CLUSTER_STATE_ALPHAMASKED) != 0)
          {
            useAlpha = true;
          }
        }
      }
    #endif
    }

    rasterBinning(group.clusterResidentID + clusterIndex, instanceID, useAlpha, useSW, renderClusterAny);
  }
}
