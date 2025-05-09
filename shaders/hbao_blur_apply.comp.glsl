/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_image_load_formatted : require

#include "hbao.h"

layout(binding=NVHBAO_MAIN_IMG_OUT)   uniform image2D   imgOut;
layout(binding=NVHBAO_MAIN_TEX_BLUR)  uniform sampler2D texSource;

#include "hbao_blur.glsl"

//-------------------------------------------------------------------------


void main()
{
  ivec2 intCoord;
  vec2  texCoord;
  
  if (setupCoordFull(intCoord, texCoord)) return;
  
  vec2 res = BlurRun(texCoord);
  vec4 color = imageLoad(imgOut, intCoord);
  imageStore(imgOut, intCoord, vec4( vec3(color.xyz * res.x), 1));
}
