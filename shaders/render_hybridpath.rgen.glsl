/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Hybrid path: the path tracer of `render_pathtrace.rgen.glsl` starting at the
  rasterized primary hit (`render_hybridpath_hit.glsl`) instead of a traced
  primary ray. Compiled with USE_HYBRIDPATH.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#include "render_pathtrace_raygen.glsl"
