/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Only used for TARGETS_RAY_TRACING
  
  A fragment shader that writes the ray tracing depth into the
  framebuffers depth buffer.

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

#include "shaderio.h"

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;

void main()
{
  ivec2 coord  = ivec2(gl_FragCoord.xy);
  gl_FragDepth = imageLoad(imgRaytracingDepth, coord).x;
}