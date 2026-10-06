/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Tone maps the linear HDR color of all renderers into the displayed image.
  Runs after DLSS at target resolution. With TONEMAP_HBAO it also composites
  HBAO (`hbao_apply.glsl`), which then skips its own apply pass.

  One pixel per 16x16 tile accumulates its log-luminance into the readback,
  from which the host derives the auto-exposure of the next frames.

  The visibility buffer and the simple sky visualizations are copied as-is,
  physically lit palette visualizations use the clip operator (`tonemapMode`).
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float : require

#include "shaderio.h"
#include "nvshaders/tonemap_functions.h.slang"

#ifndef TONEMAP_HBAO
#define TONEMAP_HBAO 0
#endif

layout(local_size_x = TONEMAP_WORKGROUP, local_size_y = TONEMAP_WORKGROUP) in;

#if TONEMAP_HBAO
#include "hbao.h"

layout(scalar, binding = BINDINGS_TONEMAP_HBAO_UBO, set = 0) uniform hbaoBuffer
{
  NVHBAOData g_Ssao;
};
layout(binding = BINDINGS_TONEMAP_HBAO_DEPTHARRAY, set = 0) uniform sampler2DArray texDepthArray;
layout(binding = BINDINGS_TONEMAP_HBAO_RESULTARRAY, set = 0) uniform sampler2DArray texResultArray;

#include "hbao_apply.glsl"
#endif

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(binding = BINDINGS_TONEMAP_IN, set = 0, rgba16f) uniform readonly image2D imgIn;
layout(binding = BINDINGS_TONEMAP_OUT, set = 0, rgba8) uniform writeonly image2D imgOut;

void main()
{
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size  = imageSize(imgIn);

#if TONEMAP_HBAO
  // before any early out, the blurred variant synchronizes the workgroup
  float occlusion = hbaoOcclusion(gl_WorkGroupID.xy, gl_LocalInvocationID.xy, gl_GlobalInvocationID.xy);
#endif

  if(any(greaterThanEqual(pixel, size)))
    return;

  vec3 color = imageLoad(imgIn, pixel).xyz;
#if TONEMAP_HBAO
  color *= occlusion;
#endif

  if(view.tonemapMode == TONEMAP_MODE_BYPASS)
  {
    imageStore(imgOut, pixel, vec4(color, 1));
    return;
  }

  if((pixel.x & 15) == 0 && (pixel.y & 15) == 0)
  {
    bool metered = true;
    if(view.tonemapper.enableCenterMetering != 0)
    {
      // only sample a centered box of the given relative size
      vec2 centered = abs(vec2(pixel) / vec2(size) * 2.0 - 1.0);
      metered       = max(centered.x, centered.y) <= view.tonemapper.centerMeteringSize;
    }
    if(metered)
    {
      atomicAdd(readback.autoExposureLumaSum, log2(max(bt709Luminance(color), 1e-3)));
      atomicAdd(readback.autoExposureSampleCount, 1u);
    }
  }

  TonemapperData tm = view.tonemapper;
  if(view.tonemapMode == TONEMAP_MODE_CLIP)
  {
    tm.method = TONEMAP_METHOD(eClip);
  }
  if(view.visualize != VISUALIZE_SHADED)
  {
    tm.contrast *= view.debugContrast;
    // grey would only saturate the sky's tint
    if(view.visualize != VISUALIZE_GREY)
      tm.saturation *= view.debugSaturation;
  }
  vec3 mapped = tm.isActive != 0 ? applyTonemap(tm, color, vec2(pixel), vec2(size)) : toSrgb(clamp(color, vec3(0), vec3(1)));
  imageStore(imgOut, pixel, vec4(mapped, 1));
}
