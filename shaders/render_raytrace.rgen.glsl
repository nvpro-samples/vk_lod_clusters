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

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba8) uniform image2D imgColor;

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;

//////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT RayPayload rayHit;

//////////////////////////////////////////////////////////////

void main()
{
  // for writing debugging values to stats.debug etc.
  bool center = gl_LaunchIDEXT.xy == (gl_LaunchSizeEXT.xy / 2);

  ivec2 screen = ivec2(gl_LaunchIDEXT.xy);
  vec2  uv     = (vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);


  vec2 d = uv * 2.0 - 1.0;


  vec4 origin    = view.viewMatrixI * vec4(0, 0, 0, 1);
  vec4 target    = normalize(view.projMatrixI * vec4(d.x, d.y, 1, 1));
  vec4 direction = normalize(view.viewMatrixI * vec4(target.xyz, 0));

  float tMin = view.nearPlane;
  float tMax = view.farPlane;

#if DEBUG_VISUALIZATION
  vec2 uvOffset            = (vec2(gl_LaunchIDEXT.xy) + vec2(1.5, 1.5)) / vec2(gl_LaunchSizeEXT.xy);
  vec2 dOffset             = uvOffset * 2.0 - 1.0;
  vec4 targetOffsetX       = normalize(view.projMatrixI * vec4(dOffset.x, d.y, 1, 1));
  vec4 targetOffsetY       = normalize(view.projMatrixI * vec4(d.x, dOffset.y, 1, 1));
  vec4 directionOffsetX    = normalize(view.viewMatrixI * vec4(targetOffsetX.xyz, 0));
  vec4 directionOffsetY    = normalize(view.viewMatrixI * vec4(targetOffsetY.xyz, 0));
  rayHit.color.xyz         = directionOffsetX.xyz;
  rayHit.differentialY.xyz = directionOffsetY.xyz;
#endif

  traceRayEXT(asScene, gl_RayFlagsCullBackFacingTrianglesEXT, 0xff, 0, 0,  // hit offset, hit stride
              0,                                                           // miss offset
              origin.xyz, tMin, direction.xyz, tMax,
              0  // rayPayloadNV location qualifier
  );

  {
    imageStore(imgColor, screen, vec4(rayHit.color.xyz, 1));
    imageStore(imgRaytracingDepth, screen, vec4(rayHit.color.w == 0 ? 1.0 : rayHit.color.w, 0.f, 0.f, 0.f));
  }
}