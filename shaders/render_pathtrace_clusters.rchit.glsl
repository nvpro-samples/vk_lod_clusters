/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Minimal closest-hit for the basic path tracer. Unlike the regular
  ray-trace hit shader it does NO shading: it only reports the hit (instance,
  cluster and triangle IDs + barycentrics + hitT). All shading happens in
  render_pathtrace.rgen.glsl.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#extension GL_EXT_spirv_intrinsics : require

// at the time of writing, no GLSL extension was available, we leverage
// GL_EXT_spirv_intrinsics to hook up the new builtin.
spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

#include "shaderio.h"

/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

layout(location = 0) rayPayloadInEXT PathRayPayload rayHit;

/////////////////////////////////

void main()
{
  rayHit.hitT       = gl_HitTEXT;
  rayHit.instanceID = gl_InstanceID;
  rayHit.clusterID  = uint(gl_ClusterIDNV_);
  rayHit.triangleID = gl_PrimitiveID;
  rayHit.bary       = barycentrics;
}
