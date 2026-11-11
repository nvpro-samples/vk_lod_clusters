/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  The path tracer of `render_pathtrace_raygen.glsl` as a compute shader with
  ray queries (--rayquery), for plain path tracing and the hybrid path.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require
#extension GL_EXT_ray_tracing_position_fetch : require
#extension GL_NV_cluster_acceleration_structure : require

#define USE_RAYQUERY 1
#define RT_LAUNCH_ID gl_GlobalInvocationID.xy
#define RT_LAUNCH_SIZE uvec2(view.viewport)

#include "render_pathtrace_raygen.glsl"

layout(local_size_x = RAYQUERY_WORKGROUP, local_size_y = RAYQUERY_WORKGROUP) in;
