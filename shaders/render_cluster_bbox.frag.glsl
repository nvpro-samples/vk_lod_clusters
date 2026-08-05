/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

///////////////////////////////////////////////////


layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

///////////////////////////////////////////////////

layout(location=0) in Interpolants
{
  flat uint clusterID;
  flat uint isPickedHit;
} IN;

///////////////////////////////////////////////////

layout(location=0,index=0) out vec4 out_Color;

///////////////////////////////////////////////////

void main()
{
  vec4 baseColor = unpackUnorm4x8(murmurHash(IN.clusterID)) * 0.9 + 0.1;
  baseColor.w    = 1.0;

  if(IN.isPickedHit != 0u)
  {
    float pulse     = 0.5 + 0.5 * sin(view.timeSec * 4.7);
    vec4  highlight = vec4(1.0, 1.0, 0.2, 1.0);
    out_Color       = mix(baseColor, highlight, pulse);
  }
  else
  {
    out_Color = baseColor;
  }
}