/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Only used for TARGETS_RASTERIZATION
  
  A fragment shader that writes the color and depth that was encoded within
  a u64 image used in compute-shader based rasterization.

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
#extension GL_EXT_shader_image_int64 : enable

#include "shaderio.h"

layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;

layout(location = 0, index = 0) out vec4 out_Color;

void main()
{
  ivec2 coord  = ivec2(gl_FragCoord.xy);
  uvec2 loaded = unpackUint2x32(imageLoad(imgRasterAtomic, coord).x);
  gl_FragDepth = loaded.y == uint(0) ? 0.0 : uintBitsToFloat(loaded.y);
  out_Color    = loaded.y == uint(0) ? vec4(0,0,0,1) : unpackUnorm4x8(loaded.x);
}