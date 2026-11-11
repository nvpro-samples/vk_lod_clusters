/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Only used for TARGETS_RAY_TRACING && USE_HYBRIDPATH && USE_HYBRID_SW_RASTER

  Writes what `render_raster_clusters_sw.comp.glsl` rasterized in this pass into the
  visibility buffer and depth, the depth test resolves it against the hardware raster.
  The list index is only valid until the next pass refills the list. Resets the pixel
  for that pass.

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
#extension GL_EXT_shader_image_int64 : enable

#include "shaderio.h"

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;

layout(location = 0, index = 0) out uvec2 out_VisibilityBuffer;

void main()
{
  ivec2 coord  = ivec2(gl_FragCoord.xy);
  uvec2 loaded = unpackUint2x32(imageLoad(imgRasterAtomic, coord).x);
  if(loaded.y == 0)
  {
    discard;
    return;
  }
  imageStore(imgRasterAtomic, coord, u64vec4(0));

  ClusterInfo cinfo    = build.renderClusterInfosSW.d[loaded.x >> HYBRIDPATH_VISBUFFER_TRIANGLE_BITS];
  uint        triangle = loaded.x & ((1u << HYBRIDPATH_VISBUFFER_TRIANGLE_BITS) - 1u);

  gl_FragDepth         = uintBitsToFloat(loaded.y);
  out_VisibilityBuffer = uvec2(cinfo.instanceID, (cinfo.clusterID << HYBRIDPATH_VISBUFFER_TRIANGLE_BITS) | triangle);
}
