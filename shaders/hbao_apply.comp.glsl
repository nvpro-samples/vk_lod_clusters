/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef NVHBAO_BLUR
#define NVHBAO_BLUR 0
#endif

#if NVHBAO_BLUR
layout(local_size_x = 16, local_size_y = 16) in;
#else
layout(local_size_x = 8, local_size_y = 8) in;
#endif

#include "hbao_apply.glsl"

void main()
{
  float totalOcclusion = hbaoOcclusion(gl_WorkGroupID.xy, gl_LocalInvocationID.xy, gl_GlobalInvocationID.xy);
  ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);

  if (all(lessThan(storePos, g_Ssao.view.viewportSize.xy)))
  {
    vec4 color = imageLoad(imgOut, storePos);
    color.xyz *= totalOcclusion;
    imageStore(imgOut, storePos, color);
  }
}
