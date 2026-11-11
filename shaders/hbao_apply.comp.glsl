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
layout(binding=NVHBAO_MAIN_TEX_DEPTH) uniform sampler2D texDepth;
layout(binding=NVHBAO_MAIN_TEX_RESULT) uniform sampler2D texResult;

layout(local_size_x = 8, local_size_y = 8) in;

#include "hbao_apply.glsl"

void main()
{
  ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);

  if (all(lessThan(storePos, g_Ssao.view.viewportSize.xy)))
  {
    float totalOcclusion = hbaoVisibility(storePos);
    vec4 color = imageLoad(imgOut, storePos);
    color.xyz *= totalOcclusion;
    imageStore(imgOut, storePos, color);
  }
}
