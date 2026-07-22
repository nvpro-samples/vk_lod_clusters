/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Miss shader for the basic path tracer. There are two payload variants:
    - index 0: primary/indirect rays. On a miss we only flag it (hitT < 0); the
      environment (physical sky) radiance is added in the ray-generation shader,
      which owns the ray direction.
    - index 1: shadow/visibility rays carrying a single float. A miss means the
      light is unoccluded, so we clear it to 0 (see traceShadowRay()).

*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#include "shaderio.h"

/////////////////////////////////

#if RAYTRACING_PAYLOAD_INDEX == 0
layout(location = RAYTRACING_PAYLOAD_INDEX) rayPayloadInEXT PathRayPayload rayHit;
#else
layout(location = RAYTRACING_PAYLOAD_INDEX) rayPayloadInEXT float rayHit;
#endif

/////////////////////////////////

void main()
{
#if RAYTRACING_PAYLOAD_INDEX == 0
  rayHit.hitT = -1.0;  // miss sentinel
#else
  rayHit = 0;
#endif
}
