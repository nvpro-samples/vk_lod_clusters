/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// based on
// https://github.com/NVIDIA-RTX/Donut/blob/main/shaders/passes/ssao_blur_cs.hlsl

// The 16 slices back into one image at AO size through a depth-aware 4x4 box, one pixel of every
// slice so the 16 noise rotations average out. Writes (visibility, view depth), the latter is what
// the upsample in `hbao_apply.glsl` compares against.

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require

#include "hbao.h"

layout(scalar, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData g_Ssao;
};

layout(binding=NVHBAO_MAIN_TEX_DEPTHARRAY)  uniform sampler2DArray texDepthArray;
layout(binding=NVHBAO_MAIN_TEX_RESULTARRAY) uniform sampler2DArray texResultArray;
layout(binding=NVHBAO_MAIN_IMG_RESULT,rg16f) uniform image2D imgResult;

layout(local_size_x = 16, local_size_y = 16) in;

const int kTile   = 24;
const int kMargin = 4;

shared vec2 s_DepthAndSSAO[kTile][kTile];

// positions clamp to the AO image: the images are allocated for full size, a downscaled run only fills their corner
vec2 loadDepthAndOcclusion(ivec2 aoPixel)
{
  ivec2 p = clamp(aoPixel, ivec2(0), g_Ssao.aoSize - 1);
  ivec3 q = ivec3(p >> 2, (p.x & 3) + (p.y & 3) * 4);
  return vec2(texelFetch(texDepthArray, q, 0).x, texelFetch(texResultArray, q, 0).x);
}

void main()
{
  ivec2 pixelPos = ivec2(gl_GlobalInvocationID.xy);

  float pixelDepth     = 0;
  float totalOcclusion = 0;

  // uniform, so the barrier stays in uniform control flow
  if(g_Ssao.blur != 0)
  {
    ivec2 tileBase = ivec2(gl_WorkGroupID.xy) * 16 - kMargin;
    for(int i = int(gl_LocalInvocationIndex); i < kTile * kTile; i += 256)
    {
      ivec2 t                  = ivec2(i % kTile, i / kTile);
      s_DepthAndSSAO[t.y][t.x] = loadDepthAndOcclusion(tileBase + t);
    }

    barrier();

    ivec2 local = ivec2(gl_LocalInvocationID.xy);
    pixelDepth  = s_DepthAndSSAO[local.y + kMargin][local.x + kMargin].x;
    if(pixelDepth > 0)
    {
      float rcpPixelDepth = 1.0 / pixelDepth;
      float totalWeight   = 0;

      // tile offsets 3..6 are pixels -1..+2 around this one: 4x4, one of every slice
      for(int y = kMargin - 1; y <= kMargin + 2; y++)
      {
        for(int x = kMargin - 1; x <= kMargin + 2; x++)
        {
          vec2  sampleDAO = s_DepthAndSSAO[local.y + y][local.x + x];
          float weight = sampleDAO.x > 0 ? clamp(1.0 - abs(pixelDepth - sampleDAO.x) * rcpPixelDepth * g_Ssao.radiusWorld, 0, 1) : 0;
          totalOcclusion += sampleDAO.y * weight;
          totalWeight += weight;
        }
      }
      totalOcclusion /= max(totalWeight, 1e-6);  // the centre always weighs 1
    }
  }
  else if(all(lessThan(pixelPos, g_Ssao.aoSize)))
  {
    vec2 dao       = loadDepthAndOcclusion(pixelPos);
    pixelDepth     = dao.x;
    totalOcclusion = dao.y;
  }

  if(all(lessThan(pixelPos, g_Ssao.aoSize)))
    imageStore(imgResult, pixelPos, vec4(pow(clamp(1.0 - totalOcclusion, 0, 1), g_Ssao.powerExponent), pixelDepth, 0, 0));
}
