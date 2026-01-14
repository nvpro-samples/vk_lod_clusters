/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION
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
  vec2 invQuantizedGbufferSize;

  float amount;
  float invBackgroundViewDepth;
  float radiusWorld;
  float surfaceBias;

  float radiusToScreen;
  float powerExponent;
};

#ifdef __cplusplus
}
#endif

#endif
