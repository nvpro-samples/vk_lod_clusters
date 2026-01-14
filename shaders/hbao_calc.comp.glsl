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

/* 
Originally based on DeinterleavedTexturing sample by Louis Bavoil
https://github.com/NVIDIAGameWorks/D3DSamples/tree/master/samples/DeinterleavedTexturing
Later version by Alexey Panteleev
https://github.com/NVIDIA-RTX/Donut/blob/main/shaders/passes/ssao_compute_cs.hlsl
*/

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require

#include "hbao.h"

layout(scalar, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData g_Ssao;
};

layout(binding=NVHBAO_MAIN_TEX_DEPTHARRAY)   uniform sampler2DArray texLinearDepth;
layout(binding=NVHBAO_MAIN_TEX_VIEWNORMAL)   uniform sampler2D      texViewNormal;
layout(binding=NVHBAO_MAIN_IMG_RESULTARRAY,r8) uniform image2DArray imgResultArray;

// Set of samples with distance from center increasing linearly,
// and angle also increasing linearly with a step of 4.5678 radians.
// Plotted on x-y dimensions, it looks pretty much random, but is intended
// to make more samples closer to the center because they have greater weight.
const vec2 g_SamplePositions[16] = vec2[](
  vec2(-0.016009523, -0.10995169),
  vec2(-0.159746436, 0.047527402),
  vec2(0.09339819, 0.201641995),
  vec2(0.232600698, 0.151846663),
  vec2(-0.220531935, -0.24995355),
  vec2(-0.251498143, 0.29661971),
  vec2(0.376870668, .23558303),
  vec2(0.201175979, 0.457742532),
  vec2(-0.535502966, -0.147913991),
  vec2(-0.076133435, 0.606350138),
  vec2(0.666537538, 0.013120791),
  vec2(-0.118107615, -0.712499494),
  vec2(-0.740973793, 0.236423582),
  vec2(0.365057451, .749117816),
  vec2(0.734614792, 0.500464349),
  vec2(-0.638657704, -0.695766948)
);

// Blue noise
const float g_RandomValues[16] = float[](
    0.059, 0.529, 0.176, 0.647,
    0.765, 0.294, 0.882, 0.412,
    0.235, 0.706, 0.118, 0.588,
    0.941, 0.471, 0.824, 0.353
);

// V = unnormalized vector from center pixel to current sample
// N = normal at center pixel
float ComputeAO(vec3 V, vec3 N, float InvR2)
{
  float VdotV = dot(V, V);
  float NdotV = dot(N, V) * inversesqrt(VdotV);
  float lambertian = clamp(NdotV - g_Ssao.surfaceBias, 0, 1);
  float falloff = clamp(1 - VdotV * InvR2, 0, 1);
  return clamp(lambertian * falloff * g_Ssao.amount, 0, 1);
}

vec2 WindowToClip(vec2 windowPos)
{
  return (windowPos.xy + g_Ssao.view.pixelOffset) * g_Ssao.view.windowToClipScale.xy + g_Ssao.view.windowToClipBias.xy;
}

vec3 ViewDepthToViewPos(vec2 clipPosXY, float viewDepth)
{
  return vec3(clipPosXY * g_Ssao.clipToView.xy * viewDepth, viewDepth);
}


#define M_PI 3.14159265f

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
  uvec3 globalId = gl_GlobalInvocationID;

  int sliceIndex = int(globalId.z);
  ivec2 sliceOffset = ivec2(sliceIndex & 3, sliceIndex >> 2);

  ivec2 pixelPos = (ivec2(globalId.xy) << 2) + sliceOffset;
  ivec2 quarterResPixelPos = pixelPos >> 2;

  float pixelViewDepth = texelFetch(texLinearDepth, ivec3(quarterResPixelPos, sliceIndex), 0).x;
  vec3 pixelNormal     = texelFetch(texViewNormal, ivec2(pixelPos), 0).xyz * 2 - 1;

  pixelNormal = normalize(pixelNormal);

  vec2 pixelClipPos = WindowToClip(pixelPos);
  vec3 pixelViewPos = ViewDepthToViewPos(pixelClipPos.xy, pixelViewDepth);

  float radiusWorld = g_Ssao.radiusWorld * max(1.0, pixelViewDepth * g_Ssao.invBackgroundViewDepth);
  float radiusPixels = radiusWorld * g_Ssao.radiusToScreen / pixelViewDepth;

  float result = 0;

  if (radiusPixels > 1)
  {
    float invRadiusWorld2 = 1.0f / (radiusWorld * radiusWorld);

    float angle = g_RandomValues[(pixelPos.x & 3) + ((pixelPos.y & 3) << 2)] * M_PI;
    vec2 sincos = vec2(sin(angle), cos(angle));

    int numSamples = 16;
    float numValidSamples = 0;

    [[unroll]]
    for (int nSample = 0; nSample < numSamples; nSample++)
    {
      vec2 sampleOffset = g_SamplePositions[nSample];
      sampleOffset = vec2(
          sampleOffset.x * sincos.y - sampleOffset.y * sincos.x, 
          sampleOffset.x * sincos.x + sampleOffset.y * sincos.y);

      vec2 sampleWindowPos = pixelPos + sampleOffset * radiusPixels + 0.5;
      ivec2 sampleWindowPosInt = ivec2(floor(sampleWindowPos * 0.25));

      float sampleViewDepth = texelFetch(texLinearDepth, ivec3(sampleWindowPosInt, sliceIndex), 0).x;
      vec2 actualClipPos = WindowToClip(vec2(sampleWindowPosInt) * 4.0 + sliceOffset + 0.5);

      if (sampleViewDepth > 0 && any(not(equal(quarterResPixelPos, sampleWindowPosInt))) && all(lessThan(abs(actualClipPos.xy), vec2(1.0))))
      {
        vec3 sampleViewPos = ViewDepthToViewPos(actualClipPos, sampleViewDepth);
        vec3 pixelToSample = sampleViewPos - pixelViewPos;
        float AO = ComputeAO(pixelToSample, pixelNormal, invRadiusWorld2);
        result += AO;
        numValidSamples += 1;
      }
    }

    if (numValidSamples > 0)
    {
      result /= numValidSamples;
    }
  }
  
  imageStore(imgResultArray, ivec3(quarterResPixelPos.xy, sliceIndex), vec4(result));
}
