/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

/*
  
  Shader Description
  ==================
  
  This compute shader does basic operations on a single thread.
  For example clamping atomic counters back to their limits or
  setting up indirect dispatches or draws etc.
  
  BUILD_SETUP_... are enums for the various operations

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
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint setup;
} push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_RENDERMATERIALS_SSBO, set = 0) buffer renderMaterialsBuffer
{
  RenderMaterial materials[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

////////////////////////////////////////////

layout(local_size_x=1) in;

////////////////////////////////////////////

#ifndef MESHSHADER_BBOX_COUNT
#define MESHSHADER_BBOX_COUNT 8
#endif


#if USE_TWO_PASS_CULLING
void setupSecondPass()
{
  // setup second pass  
  buildRW.cullPass = 1;
  buildRW.traversalTaskCounter = 0;
  buildRW.traversalGroupWriteCounter = 0;
  buildRW.traversalNodeReadCounter = 0;
  buildRW.traversalNodeWriteCounter = 0;
  buildRW.renderClusterCounter = 0;
  buildRW.renderClusterCounterSW = 0;
  buildRW.renderClusterCounterAlpha = 0;
  buildRW.renderClusterCounterAlphaSW = 0;
}
#endif

void main()
{  
  // special operations for setting up indirect dispatches
  // or clamping other operations to actual limits
  
  if (push.setup == BUILD_SETUP_TRAVERSAL_RUN)
  {
    // during traversal_init we might overshoot the traversalTaskCounter  
    uint nodeCount = min(buildRW.traversalNodeWriteCounter, build.maxTraversalInfos);
    // also set up the initial writeCounter to be equal, so that new jobs are enqueued after it
    buildRW.traversalNodeWriteCounter = nodeCount;
  #if USE_PERSISTENT_TRAVERSAL_KERNEL
    buildRW.traversalTaskCounter = int(nodeCount);
  #else
    buildRW.traversalNodeStart  = 0;
    buildRW.traversalNodeEnd    = nodeCount;
    buildRW.traversalGroupStart = 0;
    buildRW.traversalGroupEnd   = 0;

    //readback.debugA[0] = nodeCount;
    //readback.debugB[0] = 0;
    //readback.debugC[0] = 1;

    uint nodeGridCount  = (nodeCount + TRAVERSAL_RUN_WORKGROUP - 1) / TRAVERSAL_RUN_WORKGROUP;
    #if USE_16BIT_DISPATCH
      uvec3 nodeGrid = fit16bitLaunchGrid(nodeGridCount);
      buildRW.indirectDispatchNodes.gridX = nodeGrid.x;
      buildRW.indirectDispatchNodes.gridY = nodeGrid.y;
      buildRW.indirectDispatchNodes.gridZ = nodeGrid.z;
    #else
      buildRW.indirectDispatchNodes.gridX = nodeGridCount;
      buildRW.indirectDispatchNodes.gridY = 1;
      buildRW.indirectDispatchNodes.gridZ = 1;
    #endif

  #endif
  }
#if !USE_PERSISTENT_TRAVERSAL_KERNEL
  else if (push.setup == BUILD_SETUP_TRAVERSAL_RUN_PASS_COMBINED || push.setup == BUILD_SETUP_TRAVERSAL_RUN_PASS_NODES_ONLY)
  {
    uint pass = buildRW.traversalPass + 1;
    buildRW.traversalPass = pass;

    // begin at last end, end at current
    uint nodeStart = min(buildRW.traversalNodeEnd, build.maxTraversalInfos);
    uint nodeEnd   = min(buildRW.traversalNodeWriteCounter, build.maxTraversalInfos);
    uint nodeCount = nodeEnd - nodeStart;
    buildRW.traversalNodeStart = nodeStart;
    buildRW.traversalNodeEnd   = nodeEnd;

    uint groupStart = min(buildRW.traversalGroupEnd, build.maxTraversalInfos);
    uint groupEnd   = min(buildRW.traversalGroupWriteCounter, build.maxTraversalInfos);
    uint groupCount = groupEnd - groupStart;
    buildRW.traversalGroupStart = groupStart;
    buildRW.traversalGroupEnd   = groupEnd;
    if (push.setup == BUILD_SETUP_TRAVERSAL_RUN_PASS_NODES_ONLY)
    {
      groupCount = 0;
      buildRW.traversalGroupStart = 0;
      buildRW.traversalGroupEnd   = 0;
    }

    uint nodeGridCount  = (nodeCount + TRAVERSAL_RUN_WORKGROUP - 1) / TRAVERSAL_RUN_WORKGROUP;
    uint groupGridCount = (groupCount + TRAVERSAL_GROUPS_WORKGROUP - 1) / TRAVERSAL_GROUPS_WORKGROUP;

    //readback.debugA[pass] = nodeCount;
    //readback.debugB[pass] = groupCount;
    //readback.debugC[pass] = 1;

  #if USE_16BIT_DISPATCH
    uvec3 nodeGrid = fit16bitLaunchGrid(nodeGridCount);
    buildRW.indirectDispatchNodes.gridX = nodeGrid.x;
    buildRW.indirectDispatchNodes.gridY = nodeGrid.y;
    buildRW.indirectDispatchNodes.gridZ = nodeGrid.z;
    uvec3 groupGrid = fit16bitLaunchGrid(groupGridCount);
    buildRW.indirectDispatchGroups.gridX = groupGrid.x;
    buildRW.indirectDispatchGroups.gridY = groupGrid.y;
    buildRW.indirectDispatchGroups.gridZ = groupGrid.z;
  #else
    buildRW.indirectDispatchNodes.gridX = nodeGridCount;
    buildRW.indirectDispatchNodes.gridY = 1;
    buildRW.indirectDispatchNodes.gridZ = 1;
    buildRW.indirectDispatchGroups.gridX = groupGridCount;
    buildRW.indirectDispatchGroups.gridY = 1;
    buildRW.indirectDispatchGroups.gridZ = 1;
  #endif
  }
#endif
#if TARGETS_RASTERIZATION
  else if (push.setup == BUILD_SETUP_DRAW)
  {
    // during traversal_run we might overshoot visibleClusterCounter  
    uint renderClusterCounter        = buildRW.renderClusterCounter;
    uint renderClusterCounterSW      = buildRW.renderClusterCounterSW;
    uint renderClusterCounterAlpha   = buildRW.renderClusterCounterAlpha;
    uint renderClusterCounterAlphaSW = buildRW.renderClusterCounterAlphaSW;
    
    // set drawindirect for actual rendered clusters
    uint numRenderedClusters        = min(renderClusterCounter,   build.maxRenderClusters);
    uint numRenderedClustersSW      = min(renderClusterCounterSW, build.maxRenderClusters);
    uint numRenderedClustersAlpha   = min(renderClusterCounterAlpha,   build.maxRenderClusters);
    uint numRenderedClustersAlphaSW = min(renderClusterCounterAlphaSW, build.maxRenderClusters);
    
  #if USE_EXT_MESH_SHADER
    uvec3 grid = fit16bitLaunchGrid(numRenderedClusters);
    buildRW.indirectDrawClustersEXT.gridX = grid.x;
    buildRW.indirectDrawClustersEXT.gridY = grid.y;
    buildRW.indirectDrawClustersEXT.gridZ = grid.z;

    grid = fit16bitLaunchGrid(numRenderedClustersAlpha);
    buildRW.indirectDrawClustersEXT.gridX = grid.x;
    buildRW.indirectDrawClustersEXT.gridY = grid.y;
    buildRW.indirectDrawClustersEXT.gridZ = grid.z;
    
    grid = fit16bitLaunchGrid((numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
    buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
    
    grid = fit16bitLaunchGrid((numRenderedClustersAlpha + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
    buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
  #else
    buildRW.indirectDrawClustersNV.count = numRenderedClusters;
    buildRW.indirectDrawClustersNV.first = 0;

    buildRW.indirectDrawClustersAlphaNV.count = numRenderedClustersAlpha;
    buildRW.indirectDrawClustersAlphaNV.first = 0;
    
    buildRW.indirectDrawClusterBoxesNV.count = (numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT;
    buildRW.indirectDrawClusterBoxesNV.first = 0;
    
    buildRW.indirectDrawClusterBoxesAlphaNV.count = (numRenderedClustersAlpha + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT;
    buildRW.indirectDrawClusterBoxesAlphaNV.first = 0;
  #endif
    buildRW.numRenderedClusters      = numRenderedClusters;
    buildRW.numRenderedClustersAlpha = numRenderedClustersAlpha;
    
  #if USE_16BIT_DISPATCH
    uvec3 grid = fit16bitLaunchGrid(numRenderedClustersSW);
    buildRW.indirectDrawClustersSW.gridX = grid.x;
    buildRW.indirectDrawClustersSW.gridY = grid.y;
    buildRW.indirectDrawClustersSW.gridZ = grid.z;
    grid = fit16bitLaunchGrid(numRenderedClustersAlphaSW);
    buildRW.indirectDispatchClustersAlphaSW.gridX = grid.x;
    buildRW.indirectDispatchClustersAlphaSW.gridY = grid.y;
    buildRW.indirectDispatchClustersAlphaSW.gridZ = grid.z;
  #else
    buildRW.indirectDispatchClustersSW.gridX      = numRenderedClustersSW;
    buildRW.indirectDispatchClustersAlphaSW.gridX = numRenderedClustersAlphaSW;
  #endif
    buildRW.numRenderedClustersSW      = numRenderedClustersSW;
    buildRW.numRenderedClustersAlphaSW = numRenderedClustersAlphaSW;

    // keep originals for array size warnings
    // use max if there is two passes
    atomicMax(readback.numRenderClusters,   renderClusterCounter);
    atomicMax(readback.numRenderClustersSW, renderClusterCounterSW);
    atomicMax(readback.numRenderClustersAlpha,   renderClusterCounterAlpha);
    atomicMax(readback.numRenderClustersAlphaSW, renderClusterCounterAlphaSW);
    atomicMax(readback.numTraversalTasks, max(buildRW.traversalNodeWriteCounter, buildRW.traversalGroupWriteCounter));

  #if USE_RENDER_STATS
    readback.numRenderedClusters   += numRenderedClusters;
    readback.numRenderedClustersSW += numRenderedClustersSW;
    readback.numRenderedClustersAlpha += numRenderedClustersAlpha;
    readback.numRenderedClustersAlphaSW += numRenderedClustersAlphaSW;
    readback.numTraversedTasks     += buildRW.traversalNodeWriteCounter;
  #endif

    // readback.debugA[0] = numRenderedClusters;
    // readback.debugA[1] = numRenderedClustersAlpha;
    // readback.debugA[2] = numRenderedClustersSW;
    // readback.debugA[3] = numRenderedClustersAlphaSW;
  
  #if USE_TWO_PASS_CULLING
    setupSecondPass();
  #endif
  }
#endif
#if TARGETS_RAY_TRACING
  else if (push.setup == BUILD_SETUP_BLAS_INSERTION)
  {
    // during traversal_run we might overshoot visibleClusterCounter  
    uint renderClusterCounter = buildRW.renderClusterCounter;
    
    // set drawindirect for actual rendered clusters
    uint numRenderedClusters = min(renderClusterCounter, build.maxRenderClusters);
    
    buildRW.renderClusterCounter = numRenderedClusters;
  
    uint numWorkGroups =  (numRenderedClusters + BLAS_INSERT_CLUSTERS_WORKGROUP-1) / BLAS_INSERT_CLUSTERS_WORKGROUP;
  #if USE_16BIT_DISPATCH
    uvec3 grid = fit16bitLaunchGrid(numWorkGroups);  
    buildRW.indirectDispatchBlasInsertion.gridX = grid.x;
    buildRW.indirectDispatchBlasInsertion.gridY = grid.y;
    buildRW.indirectDispatchBlasInsertion.gridZ = grid.z;
  #else
    buildRW.indirectDispatchBlasInsertion.gridX = numWorkGroups;
  #endif

    // keep originals for array size warnings 
    readback.numRenderClusters = renderClusterCounter;
    readback.numTraversalTasks = max(buildRW.traversalNodeWriteCounter, buildRW.traversalGroupWriteCounter);

  #if USE_RENDER_STATS
    readback.numTraversedTasks   = buildRW.traversalNodeWriteCounter;
    readback.numRenderedClusters = numRenderedClusters;
  #endif
  
  #if USE_EXT_MESH_SHADER
    uvec3 grid = fit16bitLaunchGrid((numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);  
    buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
  #else
    buildRW.indirectDrawClusterBoxesNV.count = (numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT;
    buildRW.indirectDrawClusterBoxesNV.first = 0;
  #endif
    buildRW.numRenderedClusters = numRenderedClusters;
  }
#endif
}