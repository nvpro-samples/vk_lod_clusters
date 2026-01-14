
/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
 
// Originally based on DeinterleavedTexturing sample by Louis Bavoil
// https://github.com/NVIDIAGameWorks/D3DSamples/tree/master/samples/DeinterleavedTexturing

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require

#include "hbao.h"

layout(scalar, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData g_Ssao;
};

layout(binding=NVHBAO_MAIN_TEX_DEPTH)            uniform sampler2D  texInputDepth;
layout(binding=NVHBAO_MAIN_IMG_VIEWNORMAL,rgba8) uniform image2D    imgViewNormal;

//----------------------------------------------------------------------------------

vec3 ViewDepthToViewPos(vec2 clipPosXY, float viewDepth)
{
  return vec3(clipPosXY * g_Ssao.clipToView.xy * viewDepth, viewDepth);
}

vec3 FetchViewPos(vec2 UV)
{
  float depth = textureLod(texInputDepth,UV,0).x;
  
  vec4 clipPos = vec4(0, 0, depth, 1);
  vec4 viewPos = g_Ssao.view.matClipToView * clipPos;
  float viewDepth = viewPos.z / viewPos.w;
  
  return ViewDepthToViewPos(UV * 2 - 1, viewDepth);
}

vec3 MinDiff(vec3 P, vec3 Pr, vec3 Pl)
{
  vec3 V1 = Pr - P;
  vec3 V2 = P - Pl;
  return (dot(V1,V1) < dot(V2,V2)) ? V1 : V2;
}

vec3 ReconstructNormal(vec2 UV, vec3 P)
{
  vec3 Pr = FetchViewPos(UV + vec2(g_Ssao.view.invViewportSize.x, 0));
  vec3 Pl = FetchViewPos(UV + vec2(-g_Ssao.view.invViewportSize.x, 0));
  vec3 Pt = FetchViewPos(UV + vec2(0, g_Ssao.view.invViewportSize.y));
  vec3 Pb = FetchViewPos(UV + vec2(0, -g_Ssao.view.invViewportSize.y));
  return normalize(cross(MinDiff(P, Pr, Pl), MinDiff(P, Pt, Pb)));
}

//----------------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;
void main() {
  ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
  
  vec2 UV = vec2(storePos) * g_Ssao.view.invViewportSize;
  vec3 P  = FetchViewPos(UV);
  vec3 N  = ReconstructNormal(UV, P);
  
  if (all(lessThan(storePos, g_Ssao.view.viewportSize.xy)))
  {
    imageStore(imgViewNormal, storePos, vec4(vec3(N * 0.5 + 0.5),1));
  }
}
