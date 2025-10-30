/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
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

#if USE_EXT_MESH_SHADER
#extension GL_EXT_mesh_shader : require
#else
#extension GL_NV_mesh_shader : require
#endif
#extension GL_EXT_control_flow_attributes: require

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint numRenderInstances;
}
push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar,binding=BINDINGS_READBACK_SSBO,set=0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};

#endif

////////////////////////////////////////////

layout(location=0) out Interpolants {
  flat uint clusterID;
} OUT[];

////////////////////////////////////////////

#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE  32
#endif

#ifndef MESHSHADER_BBOX_COUNT
#define MESHSHADER_BBOX_COUNT 8
#endif



layout(local_size_x=MESHSHADER_WORKGROUP_SIZE) in;
layout(max_vertices=MESHSHADER_BBOX_COUNT * MESHSHADER_BBOX_VERTICES, max_primitives=MESHSHADER_BBOX_COUNT * MESHSHADER_BBOX_LINES) out;
layout(lines) out;

////////////////////////////////////////////

void writePrimitiveLineIndices(uint idx, uvec2 vertexIndices)
{
#if USE_EXT_MESH_SHADER
  gl_PrimitiveLineIndicesEXT[idx] = vertexIndices;
#else
  gl_PrimitiveIndicesNV[idx * 2 + 0] = vertexIndices.x;
  gl_PrimitiveIndicesNV[idx * 2 + 1] = vertexIndices.y;
#endif
}

void main()
{
#if USE_EXT_MESH_SHADER
  uint workGroupID = getWorkGroupIndexLinearized(gl_WorkGroupID);
#else
  uint workGroupID = gl_WorkGroupID.x;
#endif
  uint numRenderedClusters = build.numRenderedClusters;
  uint baseID   = workGroupID * MESHSHADER_BBOX_COUNT;  
  uint numBoxes = min(numRenderedClusters, baseID + MESHSHADER_BBOX_COUNT) - baseID;
  
#if USE_EXT_MESH_SHADER
  SetMeshOutputsEXT(numBoxes * MESHSHADER_BBOX_VERTICES, numBoxes * MESHSHADER_BBOX_LINES);
  if (numBoxes == 0) return;
#else
  if (gl_LocalInvocationID.x == 0)
  {
    gl_PrimitiveCountNV = numBoxes * MESHSHADER_BBOX_LINES;
  }
#endif

  const uint vertexRuns = ((MESHSHADER_BBOX_COUNT * MESHSHADER_BBOX_VERTICES) + MESHSHADER_WORKGROUP_SIZE-1) / MESHSHADER_WORKGROUP_SIZE;
  
  [[unroll]]
  for (uint32_t run = 0; run < vertexRuns; run++)
  {
    uint vert   = gl_LocalInvocationID.x + run * MESHSHADER_WORKGROUP_SIZE;
    uint box    = vert / MESHSHADER_BBOX_VERTICES;
    uint corner = vert % MESHSHADER_BBOX_VERTICES;
    
    uint boxLoad = min(box,numBoxes-1);
  
    ClusterInfo cinfo = build.renderClusterInfos.d[boxLoad + baseID];
    uint clusterID = cinfo.clusterID;
    RenderInstance instance = instances[cinfo.instanceID];
    
  #if USE_STREAMING
    Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
  #else
    Geometry geometry = geometries[instance.geometryID];
    Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
  #endif
    
    BBox bbox = Cluster_getBBox(clusterRef);
    
    bvec3 weight   = bvec3((corner & 1) != 0, (corner & 2) != 0, (corner & 4) != 0);
    vec3 cornerPos = mix(bbox.lo, bbox.hi, weight);
    
    if (box < numBoxes)
    {
    #if USE_EXT_MESH_SHADER
      gl_MeshVerticesEXT[vert].gl_Position = 
    #else
      gl_MeshVerticesNV[vert].gl_Position = 
    #endif
        view.viewProjMatrix * (instance.worldMatrix * vec4(cornerPos,1));
      OUT[vert].clusterID = clusterID;
    }
  }
  
  {
    uvec2 boxIndices[4] = uvec2[4](
      uvec2(0,1),uvec2(1,3),uvec2(3,2),uvec2(2,0)
    );
  
    uint subID = gl_LocalInvocationID.x & (MESHSHADER_BBOX_THREADS-1);
    uint box   = gl_LocalInvocationID.x / MESHSHADER_BBOX_THREADS;
  
    uvec2 circle = boxIndices[subID];
    
    if (box < numBoxes)
    {  
      // lower
      writePrimitiveLineIndices(box * 12 + subID + 0, circle + box * MESHSHADER_BBOX_VERTICES);
      // upper
      writePrimitiveLineIndices(box * 12 + subID + 4, circle + 4 + box * MESHSHADER_BBOX_VERTICES);
      // connectors
      writePrimitiveLineIndices(box * 12 + subID + 8, uvec2(subID, subID + 4) + box * MESHSHADER_BBOX_VERTICES);
    }
  }
}
