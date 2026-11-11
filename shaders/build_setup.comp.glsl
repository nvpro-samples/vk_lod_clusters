/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#if USE_TWO_PASS_REJECT_LISTS
  // `rejectClusterMasks` is indexed by the first pass' group slot, grab the count
  // before the counter is reset below
  uint pass0Groups = min(buildRW.traversalGroupWriteCounter, build.maxTraversalInfos);
#endif

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

#if USE_TWO_PASS_REJECT_LISTS
  // The second pass does not traverse from the roots again, it continues from what
  // the first pass rejected. Point the traversal queues at the reject arrays and
  // seed their counters, the traversal kernels themselves stay unchanged.
  // `traversal_init` appends the roots of revived instances after the reject nodes,
  // BUILD_SETUP_TRAVERSAL_RUN then derives the grids from the combined counters.
  uint rejectNodes     = min(buildRW.rejectNodeCounter, build.maxTraversalInfos);
  uint rejectGroups    = min(buildRW.rejectGroupCounter, build.maxTraversalInfos);
  uint rejectInstances = min(buildRW.rejectInstanceCounter, build.numRenderInstances);

  buildRW.pass0GroupCount = pass0Groups;

  buildRW.traversalNodeInfos  = build.rejectNodeInfos;
  buildRW.traversalGroupInfos = build.rejectGroupInfos;

  buildRW.traversalNodeWriteCounter  = rejectNodes;
  buildRW.traversalGroupWriteCounter = rejectGroups;

  uint instancesGridCount = (rejectInstances + TRAVERSAL_INIT_WORKGROUP - 1) / TRAVERSAL_INIT_WORKGROUP;
  uint clustersGridCount  = (pass0Groups + TRAVERSAL_REJECT_CLUSTERS_WORKGROUP - 1) / TRAVERSAL_REJECT_CLUSTERS_WORKGROUP;

#if USE_16BIT_DISPATCH
  uvec3 instancesGrid = fit16bitLaunchGrid(instancesGridCount);
  uvec3 clustersGrid  = fit16bitLaunchGrid(clustersGridCount);
#else
  uvec3 instancesGrid = uvec3(instancesGridCount, 1, 1);
  uvec3 clustersGrid  = uvec3(clustersGridCount, 1, 1);
#endif
  buildRW.indirectDispatchRejectInstances.gridX = instancesGrid.x;
  buildRW.indirectDispatchRejectInstances.gridY = instancesGrid.y;
  buildRW.indirectDispatchRejectInstances.gridZ = instancesGrid.z;
  buildRW.indirectDispatchRejectClusters.gridX  = clustersGrid.x;
  buildRW.indirectDispatchRejectClusters.gridY  = clustersGrid.y;
  buildRW.indirectDispatchRejectClusters.gridZ  = clustersGrid.z;

  // originals, for the array size warnings
  readback.numRejectInstances = buildRW.rejectInstanceCounter;
  readback.numRejectNodes     = buildRW.rejectNodeCounter;
  readback.numRejectGroups    = buildRW.rejectGroupCounter;
#endif
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
    if (nodeCount == 0)
    {
      // no subgroup will take the task counter to zero, groups only come from traversal_init
      uint groupCount     = min(buildRW.traversalGroupWriteCounter, build.maxTraversalInfos);
      uint workGroupCount = (groupCount + TRAVERSAL_GROUPS_WORKGROUP - 1) / TRAVERSAL_GROUPS_WORKGROUP;
    #if USE_16BIT_DISPATCH
      uvec3 grid = fit16bitLaunchGrid(workGroupCount);
      buildRW.indirectDispatchGroups.gridX = grid.x;
      buildRW.indirectDispatchGroups.gridY = grid.y;
      buildRW.indirectDispatchGroups.gridZ = grid.z;
    #else
      buildRW.indirectDispatchGroups.gridX = workGroupCount;
    #endif
    }
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
  else if (push.setup == BUILD_SETUP_HIT_DEMAND_INIT)
  {
    // reset the demand-BFS queue counters; stream_hit_propagation seeds afterwards
    buildRW.hitDemandWriteCounter = 0;
    buildRW.hitDemandStart        = 0;
    buildRW.hitDemandEnd          = 0;
    buildRW.hitDemandPass         = 0;
  }
  else if (push.setup == BUILD_SETUP_HIT_DEMAND_RUN)
  {
    // first frontier == the seeds appended by stream_hit_propagation.
    // record the unclamped peak so the UI can warn when traversalNodeInfos is too
    // small (the demand BFS shares that array with the traversal).
    readback.numHitDemandTasks = buildRW.hitDemandWriteCounter;

    uint count = min(buildRW.hitDemandWriteCounter, build.maxTraversalInfos);
  #if USE_PERSISTENT_HIT_DEMAND_KERNEL
    buildRW.traversalNodeWriteCounter = count;
    buildRW.traversalNodeReadCounter  = 0;
    buildRW.traversalTaskCounter      = int(count);
  #else
    // continue appending children after the seeds (overshoot is dropped by the
    // shader's slot < maxTraversalInfos guard)
    buildRW.hitDemandWriteCounter = count;
    buildRW.hitDemandStart        = 0;
    buildRW.hitDemandEnd          = count;

    uint gridCount = (count + STREAM_HIT_DEMAND_TREE_WORKGROUP - 1) / STREAM_HIT_DEMAND_TREE_WORKGROUP;
  #if USE_16BIT_DISPATCH
    uvec3 grid = fit16bitLaunchGrid(gridCount);
    buildRW.indirectDispatchHitDemand.gridX = grid.x;
    buildRW.indirectDispatchHitDemand.gridY = grid.y;
    buildRW.indirectDispatchHitDemand.gridZ = grid.z;
  #else
    buildRW.indirectDispatchHitDemand.gridX = gridCount;
    buildRW.indirectDispatchHitDemand.gridY = 1;
    buildRW.indirectDispatchHitDemand.gridZ = 1;
  #endif
  #endif
  }
  else if (push.setup == BUILD_SETUP_HIT_DEMAND_RUN_PASS)
  {
    // advance the frontier: the next pass processes what the previous pass appended.
    buildRW.hitDemandPass = buildRW.hitDemandPass + 1;

    readback.numHitDemandTasks = max(readback.numHitDemandTasks, buildRW.hitDemandWriteCounter);

    uint start = min(buildRW.hitDemandEnd, build.maxTraversalInfos);
    uint end   = min(buildRW.hitDemandWriteCounter, build.maxTraversalInfos);
    buildRW.hitDemandStart = start;
    buildRW.hitDemandEnd   = end;

    uint count     = end - start;
    uint gridCount = (count + STREAM_HIT_DEMAND_TREE_WORKGROUP - 1) / STREAM_HIT_DEMAND_TREE_WORKGROUP;
  #if USE_16BIT_DISPATCH
    uvec3 grid = fit16bitLaunchGrid(gridCount);
    buildRW.indirectDispatchHitDemand.gridX = grid.x;
    buildRW.indirectDispatchHitDemand.gridY = grid.y;
    buildRW.indirectDispatchHitDemand.gridZ = grid.z;
  #else
    buildRW.indirectDispatchHitDemand.gridX = gridCount;
    buildRW.indirectDispatchHitDemand.gridY = 1;
    buildRW.indirectDispatchHitDemand.gridZ = 1;
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
    buildRW.indirectDrawClustersAlphaEXT.gridX = grid.x;
    buildRW.indirectDrawClustersAlphaEXT.gridY = grid.y;
    buildRW.indirectDrawClustersAlphaEXT.gridZ = grid.z;
    
    grid = fit16bitLaunchGrid((numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
    buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
    
    grid = fit16bitLaunchGrid((numRenderedClustersAlpha + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
    buildRW.indirectDrawClusterBoxesAlphaEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesAlphaEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesAlphaEXT.gridZ = grid.z;
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
    // own scope, USE_EXT_MESH_SHADER declares a `grid` in this one as well
    {
      uvec3 grid = fit16bitLaunchGrid(numRenderedClustersSW);
      buildRW.indirectDispatchClustersSW.gridX = grid.x;
      buildRW.indirectDispatchClustersSW.gridY = grid.y;
      buildRW.indirectDispatchClustersSW.gridZ = grid.z;
      grid = fit16bitLaunchGrid(numRenderedClustersAlphaSW);
      buildRW.indirectDispatchClustersAlphaSW.gridX = grid.x;
      buildRW.indirectDispatchClustersAlphaSW.gridY = grid.y;
      buildRW.indirectDispatchClustersAlphaSW.gridZ = grid.z;
    }
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
    // own scope, USE_16BIT_DISPATCH declares a `grid` in this one as well
    {
      uvec3 grid = fit16bitLaunchGrid((numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
      buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
      buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
      buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
    }
  #else
    buildRW.indirectDrawClusterBoxesNV.count = (numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT;
    buildRW.indirectDrawClusterBoxesNV.first = 0;
  #endif
    buildRW.numRenderedClusters = numRenderedClusters;
  }
#endif
#if USE_PARTITION_EARLY_OUT
  else if (push.setup == BUILD_SETUP_PARTITION_ACTIVE)
  {
    // one workgroup per PARTITION_ACTIVE_INSTANCES_WORKGROUP instances of each active block
    uint activeBlocks = buildRW.activeBlockCounter;

    // must match how `cmdLinearDispatch` lays out the linear path, because the consuming
    // kernels resolve their index through `getWorkGroupIndex`
    uint workGroups = activeBlocks * (PARTITION_INSTANCE_BLOCK_SIZE / PARTITION_ACTIVE_INSTANCES_WORKGROUP);
    uvec3 grid      = USE_16BIT_DISPATCH != 0 ? fit16bitLaunchGrid(workGroups) : uvec3(workGroups, 1, 1);

    buildRW.indirectDispatchActiveInstances.gridX = grid.x;
    buildRW.indirectDispatchActiveInstances.gridY = grid.y;
    buildRW.indirectDispatchActiveInstances.gridZ = grid.z;

    readback.numActivePartitions = buildRW.partitionActiveCounter;
    readback.partitionInvalidateAllCount       = build.partitionInvalidateAll;
    readback.numActiveInstances  = activeBlocks * PARTITION_INSTANCE_BLOCK_SIZE;
  }
#endif
#if USE_PTLAS
  else if (push.setup == BUILD_SETUP_PTLAS_OPS)
  {
    // all instances were written at init, only change the blas of those
    // instances whose blas actually changed
    uint updateCount = buildRW.ptlasUpdateCounter;
    
    if (updateCount == 0)
    {
      // a build without operations empties the ptlas, so re-write one instance with
      // the blas it already uses
      PtlasInstanceUpdate instanceUpdate;
      instanceUpdate.instanceIndex     = 0;
      instanceUpdate.instanceSbtOffset = 0;
      instanceUpdate.blasReference     = build.ptlasInstanceBlas.d[0];
      
      build.ptlasInstanceUpdates.d[0] = instanceUpdate;
      updateCount = 1;
    }
    
    PtlasOperation operation;
    operation.opType     = PTLAS_OP_UPDATE_INSTANCE;
    operation.argCount   = updateCount;
    operation.argAddress = uint64_t(build.ptlasInstanceUpdates);
    operation.argStride  = PTLAS_INSTANCE_UPDATE_SIZE;
    
    build.ptlasOperations.d[0]     = operation;
    build.ptlasOperationCount.d[0] = 1;
    
    readback.numPtlasUpdates = buildRW.ptlasUpdateCounter;
  }
#endif
}