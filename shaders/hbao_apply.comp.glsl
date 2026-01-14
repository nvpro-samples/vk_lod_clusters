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
 
// based on
// https://github.com/NVIDIA-RTX/Donut/blob/main/shaders/passes/ssao_blur_cs.hlsl

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_scalar_block_layout : require

#include "hbao.h"

layout(scalar, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData g_Ssao;
};

layout(binding=NVHBAO_MAIN_IMG_OUT)   uniform image2D imgOut;
layout(binding=NVHBAO_MAIN_TEX_DEPTHARRAY) uniform sampler2DArray texDepthArray;
layout(binding=NVHBAO_MAIN_TEX_RESULTARRAY) uniform sampler2DArray texResultArray;


void divrem(float a, float b, out float div, out float rem)
{
  a = (a + 0.5) / b;
  div = floor(a);
  rem = floor(fract(a) * b);
}

#ifndef NVHBAO_BLUR
#define NVHBAO_BLUR 0
#endif

#if NVHBAO_BLUR
// Warning: do not change the group size or shared array dimensions without a complete understanding of the data loading method.
shared vec2 s_DepthAndSSAO[24][24];
layout(local_size_x = 16, local_size_y = 16) in;
#else
layout(local_size_x = 8, local_size_y = 8) in;
#endif

void main()
{
  uvec2 groupId = gl_WorkGroupID.xy;
  uvec2 threadId = gl_LocalInvocationID.xy;
  uvec2 globalId = gl_GlobalInvocationID.xy;

#if NVHBAO_BLUR
  int linearIdx = int((threadId.y << 4) + threadId.x);

  if (linearIdx < 144)
  {
    // Rename the threads to a 3x3x16 grid where X and Y are "offsetUV" and Z is "slice"

    vec2 offsetUVf;
    float a, slice;
    divrem(linearIdx, 3.0, a, offsetUVf.x);
    divrem(a, 3.0, slice, offsetUVf.y);

    offsetUVf *= 8;
    ivec2 offsetUV = ivec2(offsetUVf);

    ivec2 pixelPos = (ivec2(groupId.xy) * 16) + offsetUV;

    vec2 UV = (vec2(pixelPos) + 1.0) * g_Ssao.invQuantizedGbufferSize.xy;

    // Load 4 pixels from each texture, overall the thread group loads a 24x24 block of pixels.
    // For a 16x16 thread group, 20x20 pixels would be enough, but it's impossible to do that with Gather.
    
    // Each Gather instruction loads 4 adjacent pixels from a deinterleaved array, and the
    // screen-space distance between those pixels is 4, not 1.

    offsetUV.x += int(slice) & 3;
    offsetUV.y += int(slice) >> 2;

    vec4 depths = textureGather(texDepthArray, vec3(UV, slice));
    vec4 occlusions = textureGather(texResultArray, vec3(UV, slice));

    s_DepthAndSSAO[offsetUV.y + 4][offsetUV.x + 0] = vec2(depths.x, occlusions.x);
    s_DepthAndSSAO[offsetUV.y + 4][offsetUV.x + 4] = vec2(depths.y, occlusions.y);
    s_DepthAndSSAO[offsetUV.y + 0][offsetUV.x + 4] = vec2(depths.z, occlusions.z);
    s_DepthAndSSAO[offsetUV.y + 0][offsetUV.x + 0] = vec2(depths.w, occlusions.w);
  }
  
  memoryBarrierShared();
  barrier();

  float totalWeight = 0;

  float totalOcclusion = 0;
  float pixelDepth = s_DepthAndSSAO[threadId.y + 4][threadId.x + 4].x;

  float rcpPixelDepth = 1.0 / (pixelDepth);

  const bool enableFilter = true;
  const int filterLeft = enableFilter ? 3 : 4;
  const int filterRight = enableFilter ? 6 : 4;
  ivec2 filterOffset;
  for (filterOffset.y = filterLeft; filterOffset.y <= filterRight; filterOffset.y++)
  {
    for (filterOffset.x = filterLeft; filterOffset.x <= filterRight; filterOffset.x++)
    {
      vec2 sampleDAO = s_DepthAndSSAO[threadId.y + filterOffset.y][threadId.x + filterOffset.x].xy;
      float sampleDepth = sampleDAO.x;
      float sampleOcclusion = sampleDAO.y;
      
      float weight = clamp(1.0 - abs(pixelDepth - sampleDepth) * rcpPixelDepth * g_Ssao.radiusWorld, 0 , 1);
      totalOcclusion += sampleOcclusion * weight;
      totalWeight += weight;
    }
  }
  
  totalOcclusion *= 1.0 / (totalWeight);
#else
  float totalOcclusion = 0;
  ivec2 quarterResPos = ivec2(globalId.xy) / 4;
  ivec2 subPos        = ivec2(globalId.xy & 3);
  int   slicePos      = subPos.y * 4 + subPos.x;
  
  totalOcclusion = texelFetch(texResultArray, ivec3(quarterResPos, slicePos), 0).x;    
#endif
  
  totalOcclusion = pow(clamp(1.0 - totalOcclusion, 0, 1), g_Ssao.powerExponent);
  ivec2 storePos = ivec2(globalId.xy);

  if (all(lessThan(storePos, g_Ssao.view.viewportSize.xy)))
  {
    vec4 color = imageLoad(imgOut, storePos);
    color.xyz *= totalOcclusion;
    imageStore(imgOut, storePos, color);
  }
}
