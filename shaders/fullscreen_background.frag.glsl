/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

/*

  Shader Description
  ==================

  Only used for TARGETS_RASTERIZATION
  
  A fragment shader that fills the background with the procedural sky.
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
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_fragment_shader_barycentric : enable

#include "shaderio.h"
#include "nvshaders/sky_functions.h.slang"

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

///////////////////////////////////////////////////

layout(location = 0, index = 0) out vec4 out_Color;

///////////////////////////////////////////////////

void main()
{
  vec2 screenPos = ((vec2(gl_FragCoord.xy) / view.viewportf) * 2.0) - 1.0;
  
  vec4 transformed = view.skyProjMatrixI * vec4(screenPos, 1.0,  1);
  vec3 rayDir      = normalize(transformed.xyz);
  
  vec3 skyColor = evalSimpleSky(view.skyParams, rayDir);

  out_Color = vec4(skyColor, 1);
}