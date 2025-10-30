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

/*
  
  Shader Description
  ==================
  
  USE_BLAS_MERGING && TARGETS_RAY_TRACING only 
  
  This compute shader writes the streaming request for
  groups to be unloaded. We determine this based on an
  age since the group has been used last.
  
  It also builds the merged instance blas based on residency
  of cluster groups. The motivation is to build a blas with
  the highest available detail.
  
  A thread represents one resident group.
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_subgroup_extended_types_int64 : enable
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

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

////////////////////////////////////////////

layout(local_size_x=TRAVERSAL_BLAS_MERGING_WORKGROUP) in;

////////////////////////////////////////////

#include "streaming.glsl"

////////////////////////////////////////////

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);

  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_AGEFILTER_CLUSTERS_WORKGROUP
  uint residentID   = streaming.resident.activeGroups.d[threadID];
  bool isValid      = threadID < streaming.resident.activeGroupsCount;

  if (isValid)
  {
    Group_in groupRef = streaming.resident.groups.d[residentID].group;
    uint geometryID   = streaming.resident.groups.d[residentID].geometryID;
    
    streamingAgeFilter(residentID, geometryID, groupRef, USE_BLAS_CACHING != 0);
    
    // if this groups' geometry requires a merged instance blas, then contribute all resident highest
    // detail clusters to it.
    uint mergedInstanceID = build.geometryBuildInfos.d[geometryID].mergedInstanceID;
    if (mergedInstanceID != ~0)
    {
      Geometry geometry  = geometries[geometryID];
      
      uint renderClusterMask = 0;
    
      for (uint clusterIndex = 0; clusterIndex < groupRef.d.clusterCount; clusterIndex++)
      {
        uint clusterGeneratingGroup = Group_getGeneratingGroup(groupRef, clusterIndex);
        
        // add clusters if we are at the highest detail, or if the generating group is not resident.
        if (clusterGeneratingGroup == SHADERIO_ORIGINAL_MESH_GROUP
          || geometry.streamingGroupAddresses.d[clusterGeneratingGroup] >= STREAMING_INVALID_ADDRESS_START)
        {
          renderClusterMask |= 1 << clusterIndex;
        }
      }
      
      // reserve space for clusters
      uint renderClusterCount = bitCount(renderClusterMask);
      uvec4 voteActive        = subgroupBallot(true);
      uint offsetClusters     = subgroupExclusiveAdd(renderClusterCount);
      uint lastActiveLane     = subgroupBallotFindMSB(voteActive);
      
      uint offsetClustersBase;
      if (gl_SubgroupInvocationID == lastActiveLane) {
        offsetClustersBase = atomicAdd(buildRW.renderClusterCounter, offsetClusters + renderClusterCount);
      }
      offsetClustersBase = subgroupShuffle(offsetClustersBase, lastActiveLane);
      offsetClusters += offsetClustersBase;
      
      // store per-thread base, to derive per-thread counts later on
      offsetClustersBase = offsetClusters;
      
      // append clusters to render list
      ClusterInfo clusterInfo;
      clusterInfo.instanceID = mergedInstanceID;
      for (uint clusterIndex = 0; clusterIndex < groupRef.d.clusterCount; clusterIndex++)
      {
        if ((renderClusterMask & (1<<clusterIndex)) != 0 && offsetClusters < build.maxRenderClusters){
          clusterInfo.clusterID  = groupRef.d.clusterResidentID + clusterIndex;
          build.renderClusterInfos.d[offsetClusters] = clusterInfo;
          offsetClusters++;
        }
      }
      
      atomicAdd(build.instanceBuildInfos.d[mergedInstanceID].clusterReferencesCount, offsetClusters - offsetClustersBase);
    }
  }
}

