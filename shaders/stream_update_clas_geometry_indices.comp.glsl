/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/


/*
  
Shader Description
==================

This compute shader fills the geometryIndexAndFlagsBuffer for the new CLAS that
need to be built for the newly-loaded clusters.

The tasks are generated inside `stream_update_scene.comp.glsl`,
look for the `useGeometryIndices` section.

Each subgroup fills one cluster's triangles in a loop. We set the
appropriate geometry index (0 or 1) based on the material of the triangle
and set the flags for opaque and two-sided triangles.

one thread represents one triangle, as we loop over all.

The kernel is launched indirectly and the grid size is computed in
`stream_setup.comp.glsl` see `STREAM_SETUP_UPDATE_GEOMETRY_INDICES`.

Inside `scene_streaming_utils.cpp` we ensure that there is enough memory for the worst
case where all triangles of all newly-loaded clusters require these unique indices
(see `StreamingUpdates::initClas`).

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

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

////////////////////////////////////////////

layout(local_size_x=STREAM_UPDATE_CLAS_GEOMETRY_INDICES_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_UPDATE_SCENE_WORKGROUP
  uint workID = getWorkGroupIndex(gl_WorkGroupID) * GEOMETRY_INDICES_TASKS_PER_WORKGROUP;
  workID += gl_SubgroupID;

  if (workID >= streaming.update.newClasGeometryIndicesTaskCounter)
  {
    return;
  }

  uint base = workID * CLUSTER_TRIANGLE_COUNT;
  uint64s_inout clusterInfos = uint64s_inout(streaming.update.newClasGeometryIndices);
  uint32s_inout geometryIndices = uint32s_inout(streaming.update.newClasGeometryIndices);
  Cluster_in clusterRef = Cluster_in(clusterInfos.d[base/2]);
  uint8s_in triMaterials = Cluster_getTriangleMaterials(Cluster_in(clusterRef));
  uint triangleCount = clusterRef.d.triangleCountMinusOne + 1;

  for (uint t = gl_SubgroupInvocationID; t < triangleCount; t += SUBGROUP_SIZE)
  {
    uint mat = uint(triMaterials.d[t]);
    uint geoIdx =
        ((mat & SHADERIO_CLUSTER_TRIANGLE_ALPHAMASKED) != 0u) ? 1u : 0u;
    uint flags = 0u;
    if ((mat & SHADERIO_CLUSTER_TRIANGLE_ALPHAMASKED) == 0u)
    {
      flags |= ClasGeometryFlag_OPAQUE_BIT_NV;
    }
    if ((mat & SHADERIO_CLUSTER_TRIANGLE_TWOSIDED) != 0u)
    {
      flags |= ClasGeometryFlag_CULL_DISABLE_BIT_NV;
    }
    geometryIndices.d[base + t] = geoIdx | flags;
  }
}