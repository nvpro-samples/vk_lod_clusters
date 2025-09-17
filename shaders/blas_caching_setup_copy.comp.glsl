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
  
  USE_BLAS_CACHING && USE_STREAMING && USE_RAY_TRACING only
  
  This compute shader sets up the copying of BLAS into cached BLAS destinations.
  
  One thread represents one streaming's cached blas patch
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
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint setup;
} push;

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

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;  
};
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

////////////////////////////////////////////

layout(local_size_x=BLAS_CACHING_SETUP_COPY_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);
  
  // guaranteed to be loadable
  StreamingGeometryPatch sgpatch = streaming.update.geometryPatches.d[threadID];
  uint64_t dstAddress            = sgpatch.cachedBlasAddress;
  
  // some patches may contain nullt address when a cached BLAS is removed completely
  if (threadID < streaming.update.patchCachedBlasCount && dstAddress != uint64_t(0))
  {
    // configure the BLAS copy operation to the persistent storage address
    
    uint buildIndex = build.geometryBuildInfos.d[sgpatch.geometryID].cachedBuildIndex;    
    uint copyOffset = atomicAdd(buildRW.cachedBlasCopyCounter, 1);
    
    build.cachedBlasClusterAddressesSrc.d[copyOffset] = build.blasBuildAddresses.d[buildIndex];
    build.cachedBlasClusterAddressesDst.d[copyOffset] = dstAddress;
  }
}