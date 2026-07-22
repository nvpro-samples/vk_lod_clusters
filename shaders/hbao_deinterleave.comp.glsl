/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
 
// based on
// https://github.com/NVIDIA-RTX/Donut/blob/main/shaders/passes/ssao_deinterleave_cs.hlsl

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require

#include "hbao.h"

layout(scalar, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData g_Ssao;
};

layout(binding=NVHBAO_MAIN_TEX_DEPTH)            uniform sampler2D      texInputDepth;
layout(binding=NVHBAO_MAIN_IMG_DEPTHARRAY,r32f)  uniform image2DArray   imgDepthArray;
layout(binding=NVHBAO_MAIN_IMG_DEPTHARRAY,rgba8) uniform image2D        imgViewNormal;

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
  ivec3 globalId  = ivec3(gl_GlobalInvocationID);
  ivec2 groupBase = globalId.xy * 4;
  
  float depths[4 * 4];

  [[unroll]]
  for (int y = 0; y < 4; y++)
  { 
    [[unroll]]
    for (int x = 0; x < 4; x++)
    {
      ivec2 gbufferSamplePos = groupBase + ivec2(x, y);
      float depth = texelFetch(texInputDepth, gbufferSamplePos, 0).x;

      vec4 clipPos = vec4(0, 0, depth, 1);
      vec4 viewPos = g_Ssao.view.matClipToView * clipPos;
      float linearDepth = -viewPos.z / viewPos.w;

      depths[y * 4 + x] = linearDepth;
    }
  }

  ivec2 quarterResPos = groupBase >> 2;
    
  [[unroll]]
  for(uint i = 0; i < 16; i++)
  {
    float depth = depths[i];
    imageStore(imgDepthArray, ivec3(quarterResPos.xy, i), vec4(depth));
  }
}
