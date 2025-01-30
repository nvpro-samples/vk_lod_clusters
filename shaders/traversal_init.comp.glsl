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

/*
  
  Shader Description
  ==================
  
  This compute shader initializes the traversal queue with the 
  root nodes of the lod hierarchy of rendered instances.

  A thread represents one instance.
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

#extension GL_NV_shader_subgroup_partitioned : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
  FrameConstants viewLast;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

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

////////////////////////////////////////////

void main()
{
  uint instanceID = gl_GlobalInvocationID.x;
  uint instanceLoad = min(build.numRenderInstances-1, instanceID);

  // TODO optimization:
  // For better loading behavior when streaming, the instances should be sorted
  // relative to camera position.
  
  RenderInstance instance = instances[instanceLoad];
  Geometry geometry = geometries[instance.geometryID];
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
  uint status = 0;
  
  bool inFrustum = intersectFrustum(geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax) && intersectHiz(clipMin, clipMax)));
  
  status  = (inFrustum ? INSTANCE_FRUSTUM_BIT : 0) |
            (isVisible ? INSTANCE_VISIBLE_BIT : 0);
  

  bool doNode = instanceID == instanceLoad 
  #if USE_CULLING && TARGETS_RASTERIZATION
    && isVisible
  #endif
    ;
  uvec4 voteNodes = subgroupBallot(doNode);
  
  // TODO optimization: enqueue all root children, so traversal can start with more nodes immediately
  // TODO feature: allow single-lod level render option by picking a single appropriate child of the root node
  // The root hierarchy node of a geometry is up to 32 wide, and each child represents one distinct lod level.
  
  uint offsetNodes = 0;
  if (subgroupElect())
  {
    offsetNodes = atomicAdd(buildRW.traversalTaskCounter, int(subgroupBallotBitCount(voteNodes)));
  }
  
  offsetNodes = subgroupBroadcastFirst(offsetNodes);  
  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);
      
  if (doNode && offsetNodes < build.maxTraversalInfos) {
    uint packedNode = geometry.nodes.d[0].packed;
    TraversalInfo traversalInfo;
    traversalInfo.instanceID = instanceID;
    traversalInfo.packedNode = packedNode;
    build.traversalNodeInfos.d[offsetNodes] = packTraversalInfo(traversalInfo);
  }

  #if TARGETS_RAY_TRACING
  if (instanceID == instanceLoad) {
    build.instanceStates.d[instanceID] = status;
    build.blasBuildInfos.d[instanceID].clusterReferencesCount = 0;
    build.blasBuildInfos.d[instanceID].clusterReferencesStride = 8;
  }
  #endif
}