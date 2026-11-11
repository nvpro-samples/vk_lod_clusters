/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVHBAO_H_
#define NVHBAO_H_

#define NVHBAO_MAIN_UBO 0
#define NVHBAO_MAIN_TEX_DEPTH 1
#define NVHBAO_MAIN_TEX_VIEWNORMAL 2
#define NVHBAO_MAIN_TEX_DEPTHARRAY 3
#define NVHBAO_MAIN_TEX_RESULTARRAY 4
#define NVHBAO_MAIN_IMG_VIEWNORMAL 5
#define NVHBAO_MAIN_IMG_DEPTHARRAY 6
#define NVHBAO_MAIN_IMG_RESULTARRAY 7
#define NVHBAO_MAIN_IMG_RESULT 8
#define NVHBAO_MAIN_TEX_RESULT 9
#define NVHBAO_MAIN_IMG_OUT 14

#ifdef __cplusplus
namespace glsl {
using namespace glm;
#endif

struct NVHBAOView
{
  mat4 matWorldToView;
  mat4 matClipToView;

  vec2  windowToClipScale;
  vec2  windowToClipBias;
  vec2  pixelOffset;
  ivec2 viewportSize;
  vec2  invViewportSize;
};

struct NVHBAOData
{
  NVHBAOView view;

  vec2 clipToView;

  float amount;
  float invBackgroundViewDepth;
  float radiusWorld;
  float surfaceBias;

  float radiusToScreen;
  float powerExponent;

  // the grid the effect runs on: viewport / downscale (1 or 2, a quarter of the pixels)
  ivec2 aoSize;
  ivec2 quarterSize;  // aoSize / 4, the slices' used size
  int   downscale;
  int   blur;
};

#ifdef __cplusplus
}
#endif

#endif
