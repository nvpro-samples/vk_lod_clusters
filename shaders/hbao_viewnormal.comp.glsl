/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
 
// Originally based on DeinterleavedTexturing sample by Louis Bavoil
// https://github.com/NVIDIAGameWorks/D3DSamples/tree/master/samples/DeinterleavedTexturing

// View normals at AO size, rebuilt around the framebuffer pixel each AO pixel sampled, from its own
// neighbours one framebuffer pixel away (with downscale the AO pixels are jittered and unevenly spaced,
// which degenerates the cross product).
//
// The tangent comes from the side whose two pixels extrapolate to this pixel's depth, not from the
// nearer one: at a crease the nearer neighbour of a grazing ground pixel is often across it on the
// wall, which gives the pixel the wall's normal and dark dashes along the crease. 1 / viewDepth is
// linear in screen space across a plane, so the side that misses less is on the same plane (after
// atyuwen, "Accurate normal reconstruction from depth buffer", 2017).

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require

#include "hbao.h"

layout(scalar, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData g_Ssao;
};

layout(binding=NVHBAO_MAIN_TEX_DEPTH)            uniform sampler2D  texInputDepth;
layout(binding=NVHBAO_MAIN_IMG_VIEWNORMAL,rgba8) uniform image2D    imgViewNormal;

#include "hbao_common.glsl"

const float kNoPlane = 1.0e30;

// 1 / view depth, 0 for the background or outside the framebuffer
float inverseDepth(ivec2 fbPixel)
{
  if(any(lessThan(fbPixel, ivec2(0))) || any(greaterThanEqual(fbPixel, g_Ssao.view.viewportSize)))
    return 0.0;
  float viewDepth = hbaoViewDepth(texelFetch(texInputDepth, fbPixel, 0).x);
  return viewDepth > 0.0 ? 1.0 / viewDepth : 0.0;
}

vec3 viewPos(ivec2 fbPixel, float invDepth)
{
  return hbaoViewPos(hbaoFramebufferToClip(fbPixel), 1.0 / invDepth);
}

// tangent along `step`, from the side that continues this pixel's plane, pointing along +step
vec3 tangent(ivec2 f, float d0, vec3 P, ivec2 step)
{
  float dPos1 = inverseDepth(f + step);
  float dPos2 = inverseDepth(f + step * 2);
  float dNeg1 = inverseDepth(f - step);
  float dNeg2 = inverseDepth(f - step * 2);

  float errPos = (dPos1 > 0.0 && dPos2 > 0.0) ? abs(2.0 * dPos1 - dPos2 - d0) : kNoPlane;
  float errNeg = (dNeg1 > 0.0 && dNeg2 > 0.0) ? abs(2.0 * dNeg1 - dNeg2 - d0) : kNoPlane;

  // no side has two surface pixels: take whichever neighbour exists
  if(errPos >= kNoPlane && errNeg >= kNoPlane)
  {
    if(dPos1 > 0.0)
      return viewPos(f + step, dPos1) - P;
    if(dNeg1 > 0.0)
      return P - viewPos(f - step, dNeg1);
    return vec3(vec2(step), 0.0);
  }

  return errPos <= errNeg ? viewPos(f + step, dPos1) - P : P - viewPos(f - step, dNeg1);
}

layout(local_size_x = 8, local_size_y = 8) in;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if(any(greaterThanEqual(p, g_Ssao.aoSize)))
    return;

  ivec2 f  = hbaoFramebufferPixel(p);
  float d0 = inverseDepth(f);
  if(d0 <= 0.0)
  {
    imageStore(imgViewNormal, p, vec4(0.5, 0.5, 0.0, 1.0));
    return;
  }

  vec3 P = viewPos(f, d0);
  vec3 N = normalize(cross(tangent(f, d0, P, ivec2(1, 0)), tangent(f, d0, P, ivec2(0, 1))));
  if(dot(N, P) > 0.0)
    N = -N; // a visible surface faces against the ray from the eye

  imageStore(imgViewNormal, p, vec4(N * 0.5 + 0.5, 1));
}
