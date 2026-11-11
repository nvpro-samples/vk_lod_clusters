/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The HBAO visibility of a framebuffer pixel, shared by `hbao_apply.comp.glsl` and the tonemap pass.
// The includer declares `g_Ssao`, `texDepth` (the depth buffer) and `texResult` (`hbao_blur.comp.glsl`).
//
// With downscale 2 a pixel falls between four AO pixels. Each is weighted by a tent of the distance
// to the pixel it sampled and by how close the view depth it was computed at is to this pixel's own
// (zero at 10% apart), so silhouettes do not bleed occlusion onto what is in front or behind. When
// none is close the closest one wins.

#include "hbao_common.glsl"

float hbaoVisibility(ivec2 pixel)
{
  float viewZ = hbaoViewDepth(texelFetch(texDepth, pixel, 0).x);
  if(viewZ <= 0.0)
    return 1.0;

  if(g_Ssao.downscale <= 1)
    return texelFetch(texResult, pixel, 0).r;

  ivec2 aoMax = g_Ssao.aoSize - 1;
  ivec2 base  = (pixel - 1) >> 1;

  float totalAo     = 0.0;
  float totalWeight = 0.0;
  float bestDiff    = 1e30;
  float bestAo      = 1.0;
  for(int i = 0; i < 4; i++)
  {
    ivec2 aoPixel  = clamp(base + ivec2(i & 1, i >> 1), ivec2(0), aoMax);
    vec2  tap      = texelFetch(texResult, aoPixel, 0).rg;
    vec2  d        = abs(vec2(hbaoFramebufferPixel(aoPixel) - pixel)) * 0.5;
    float bilinear = max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0) + 1e-3;
    // the background (view depth 0) comes out at 1 and weighs nothing
    float relDiff = abs(tap.y - viewZ) / viewZ;
    float weight  = bilinear * clamp(1.0 - relDiff * 10.0, 0.0, 1.0);
    totalAo += tap.x * weight;
    totalWeight += weight;
    if(relDiff < bestDiff)
    {
      bestDiff = relDiff;
      bestAo   = tap.x;
    }
  }
  return totalWeight > 1e-4 ? totalAo / totalWeight : bestAo;
}
