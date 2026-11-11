/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared by the HBAO passes. The includer declares `g_Ssao`.
//
// AO pixel p stands for one framebuffer pixel, a point sample: p itself, or with downscale 2 one
// pixel of the 2x2 block at 2p picked by a hash of p. A regular grid would alias thin geometry,
// the hash turns that into noise the blur absorbs. Spatial, so nothing flickers.

ivec2 hbaoQuarterJitter(ivec2 aoPixel)
{
  uint h = uint(aoPixel.x) * 0x8da6b343u ^ uint(aoPixel.y) * 0xd8163841u;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return ivec2(int(h & 1u), int((h >> 1) & 1u));
}

ivec2 hbaoFramebufferPixel(ivec2 aoPixel)
{
  ivec2 fb = g_Ssao.downscale > 1 ? aoPixel * 2 + hbaoQuarterJitter(aoPixel) : aoPixel;
  return min(fb, g_Ssao.view.viewportSize - 1);
}

// clip xy of the centre of a framebuffer pixel
vec2 hbaoFramebufferToClip(ivec2 fbPixel)
{
  fbPixel = clamp(fbPixel, ivec2(0), g_Ssao.view.viewportSize - 1);
  return (vec2(fbPixel) + 0.5) * g_Ssao.view.windowToClipScale + g_Ssao.view.windowToClipBias;
}

vec2 hbaoWindowToClip(ivec2 aoPixel)
{
  return hbaoFramebufferToClip(hbaoFramebufferPixel(aoPixel));
}

vec3 hbaoViewPos(vec2 clipPosXY, float viewDepth)
{
  return vec3(clipPosXY * g_Ssao.clipToView.xy * viewDepth, viewDepth);
}

// positive view depth, 0 for the background (reversed-Z, cleared to 0)
float hbaoViewDepth(float rawDepth)
{
  if(rawDepth <= 0.0)
    return 0.0;
  vec4 viewPos = g_Ssao.view.matClipToView * vec4(0, 0, rawDepth, 1);
  return -viewPos.z / viewPos.w;
}
