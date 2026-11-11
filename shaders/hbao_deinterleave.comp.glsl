/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
 
// based on
// https://github.com/NVIDIA-RTX/Donut/blob/main/shaders/passes/ssao_deinterleave_cs.hlsl

// view depth deinterleaved 4x4: AO pixel p lands in slice (p.x & 3) + (p.y & 3) * 4 at p / 4

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

#include "hbao_common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
  ivec2 quarterPos = ivec2(gl_GlobalInvocationID.xy);
  if(any(greaterThanEqual(quarterPos, g_Ssao.quarterSize)))
    return;

  ivec2 groupBase = quarterPos * 4;

  [[unroll]]
  for (int y = 0; y < 4; y++)
  { 
    [[unroll]]
    for (int x = 0; x < 4; x++)
    {
      ivec2 aoPixel = groupBase + ivec2(x, y);
      float depth   = 0.0;
      if(all(lessThan(aoPixel, g_Ssao.aoSize)))
        depth = hbaoViewDepth(texelFetch(texInputDepth, hbaoFramebufferPixel(aoPixel), 0).x);

      imageStore(imgDepthArray, ivec3(quarterPos, y * 4 + x), vec4(depth));
    }
  }
}
