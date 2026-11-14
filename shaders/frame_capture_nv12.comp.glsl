/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Converts a captured frame from rgba8 to nv12, bt709 limited range, for the video
  encoder of the frame capture. The input is display encoded already, so the matrix
  applies to it directly. Chroma is the average of each 2x2 block. A source of another
  size than the video is scaled to fit, nearest sample, the borders are black.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio_frame_capture.h"

#if FRAME_CAPTURE_READ_IMAGE
#extension GL_EXT_samplerless_texture_functions : require
layout(set = 0, binding = 0) uniform texture2D srcImage;
#endif

layout(push_constant, scalar) uniform pushData
{
  FrameCaptureNv12Constants push;
};

layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer Words_in
{
  uint d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) writeonly buffer Words_out
{
  uint d[];
};

layout(local_size_x = FRAME_CAPTURE_NV12_WORKGROUP_X, local_size_y = FRAME_CAPTURE_NV12_WORKGROUP_Y) in;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

uint quantize(float value, float offset, float scale)
{
  return uint(clamp(offset + scale * value + 0.5, 0.0, 255.0));
}

vec3 fetch(Words_in src, uint x, uint y)
{
  // an exact fetch at scale 1
  ivec2 s = ivec2(floor((vec2(x, y) + 0.5 - push.offset) * push.invScale));
  if(s.x < 0 || s.y < 0 || s.x >= int(push.srcWidth) || s.y >= int(push.srcHeight))
    return vec3(0);
#if FRAME_CAPTURE_READ_IMAGE
  if(push.readImage != 0)
    return texelFetch(srcImage, s, 0).rgb;
#endif
  vec3 rgb = unpackUnorm4x8(src.d[s.y * push.srcWidth + s.x]).rgb;
  return push.swapRedBlue != 0 ? rgb.bgr : rgb;
}

void main()
{
  uint x = gl_GlobalInvocationID.x * 4;
  uint y = gl_GlobalInvocationID.y * 2;
  if(x >= push.width || y >= push.height)
    return;

  Words_in  src = Words_in(push.srcAddress);
  Words_out dst = Words_out(push.dstAddress);

  vec3 sums[2] = {vec3(0), vec3(0)};
  for(uint row = 0; row < 2; row++)
  {
    uint lumaWord = 0;
    for(uint i = 0; i < 4; i++)
    {
      vec3 rgb = fetch(src, x + i, y + row);
      lumaWord |= quantize(dot(rgb, LUMA), 16.0, 219.0) << (i * 8);
      sums[i / 2] += rgb;
    }
    dst.d[((y + row) * push.width + x) / 4] = lumaWord;
  }

  uint chromaWord = 0;
  for(uint i = 0; i < 2; i++)
  {
    vec3  rgb  = sums[i] * 0.25;
    float luma = dot(rgb, LUMA);
    chromaWord |= quantize((rgb.b - luma) / 1.8556, 128.0, 224.0) << (i * 16);
    chromaWord |= quantize((rgb.r - luma) / 1.5748, 128.0, 224.0) << (i * 16 + 8);
  }
  dst.d[(push.width * push.height + (y / 2) * push.width + x) / 4] = chromaWord;
}
