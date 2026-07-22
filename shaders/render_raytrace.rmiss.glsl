/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#include "shaderio.h"
#include "nvshaders/sky_functions.h.slang"

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

//////////////////////////////////////////////////////////////

#if RAYTRACING_PAYLOAD_INDEX == 0
layout(location = RAYTRACING_PAYLOAD_INDEX) rayPayloadInEXT RayPayload rayHit;
#else
layout(location = RAYTRACING_PAYLOAD_INDEX) rayPayloadInEXT float rayHit;
#endif
//////////////////////////////////////////////////////////////

void main()
{
#if RAYTRACING_PAYLOAD_INDEX == 0
#if ALLOW_SHADING
  vec3 skyColor = evalSimpleSky(view.skyParams, gl_WorldRayDirectionEXT);
  rayHit.color = skyColor;
  rayHit.hitT  = 0;
  #if USE_DLSS
    rayHit.dlssAlbedo = vec4(skyColor,1);
    rayHit.dlssNormalRoughness = vec4(0);
    rayHit.dlssSpecular = vec3(0);
  #endif
#elif USE_DEPTH_ONLY
  rayHit.hitT  = 0;
#else
  rayHit.color = vec3(0);
  rayHit.hitT  = 0;
#endif
#else
  rayHit = 0;
#endif
}
