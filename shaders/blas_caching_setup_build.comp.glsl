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
  
  USE_BLAS_CACHING && USE_RAY_TRACING && USE_STREAMING only
  
  This compute shader handles setup of building the BLAS for blas caching.
  The streaming update contains which geometry to build.
  
  The threads cooperatively fill all cluster references for the BLAS 
  from the cluster groups of the specified lod level.
  
  One workgroup operates on one geometry that is provided through the
  streaming's cached blas patch.
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
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

////////////////////////////////////////////

layout(local_size_x=BLAS_CACHING_SETUP_BUILD_WORKGROUP) in;

////////////////////////////////////////////

shared uint s_clusterOffset;
shared uint s_buildIndex;

void main()
{
  uint patchID       = getWorkGroupIndex(gl_WorkGroupID);
  uint localThreadID = gl_LocalInvocationID.x;
  
  StreamingGeometryPatch sgpatch = streaming.update.geometryPatches.d[patchID];
  
  uint cachedBlasLodLevel      = sgpatch.cachedBlasLodLevel;
  uint cachedBlasClustersCount = sgpatch.cachedBlasClustersCount;
  uint geometryID              = sgpatch.geometryID;
  
  if (localThreadID == 0)
  {  
    geometries[geometryID].cachedBlasLodLevel = uint8_t(cachedBlasLodLevel);
    geometries[geometryID].cachedBlasAddress  = sgpatch.cachedBlasAddress;
  }
  
  if (cachedBlasLodLevel == TRAVERSAL_INVALID_LOD_LEVEL)
  {
    // rare event we fully disable the cached BLAS
    return;
  }
  
  Geometry geometry     = geometries[geometryID];
  LodLevel lodLevelInfo = geometry.lodLevels.d[cachedBlasLodLevel];
  
  if (localThreadID == 0)
  {
    // host's `SceneStreaming::handleBlasCaching` ensured that there is enough space for these
    // offsets to be always valid.
    
    uint referencesOffset = atomicAdd(buildRW.blasClasCounter, cachedBlasClustersCount);
    uint buildOffset      = atomicAdd(buildRW.blasBuildCounter, 1);
    
    build.geometryBuildInfos.d[geometryID].cachedBuildIndex = buildOffset;
    
    // setup insertion of clusters for builds
    build.blasBuildInfos.d[buildOffset].clusterReferencesCount  = cachedBlasClustersCount;
    build.blasBuildInfos.d[buildOffset].clusterReferencesStride = 8;
    build.blasBuildInfos.d[buildOffset].clusterReferences       = uint64_t(build.blasClusterAddresses) + uint64_t(referencesOffset * 8);
    
    // share across workgroup
    s_clusterOffset = referencesOffset;
    s_buildIndex    = buildOffset;
  }
  
  memoryBarrierShared();
  barrier();
  
  uint buildIndex = s_buildIndex;
  
  // iterate over all groups in this level to compute per subgroup cluster count
  uint subgroupClustersCount = 0;
  for (uint i = localThreadID; i < lodLevelInfo.groupCount; i += BLAS_CACHING_SETUP_BUILD_WORKGROUP)
  {
    Group_in groupRef = Group_in(geometry.streamingGroupAddresses.d[i + lodLevelInfo.groupOffset]);
    subgroupClustersCount += subgroupAdd(uint(groupRef.d.clusterCount));
  }
  
  // combine to running offset across the workgroup
  uint subgroupClustersOffset = 0;
  if (gl_SubgroupInvocationID == 0)
  {
    subgroupClustersOffset = atomicAdd(s_clusterOffset, subgroupClustersCount);
  }
  
  subgroupClustersOffset = subgroupShuffle(subgroupClustersOffset, 0);
  
  // insert all clusters by iterating over all groups of the cached lod level.
  for (uint i = localThreadID; i < lodLevelInfo.groupCount; i += BLAS_CACHING_SETUP_BUILD_WORKGROUP)
  {
    // these addresses are guaranteed to be valid
    Group_in groupRef  = Group_in(geometry.streamingGroupAddresses.d[i + lodLevelInfo.groupOffset]);
    
    uint clusterID     = groupRef.d.clusterResidentID;
    uint clustersCount = groupRef.d.clusterCount;
    
    subgroupClustersOffset         += subgroupExclusiveAdd(clustersCount);
    uint lastLane                   = subgroupBallotFindMSB(subgroupBallot(true));    
    uint subgroupClustersOffsetNext = subgroupShuffle(subgroupClustersOffset + clustersCount, lastLane);
    
    for (uint c = 0; c < clustersCount; c++)
    {
      uint clusterIndex = subgroupClustersOffset + c;
      build.blasClusterAddresses.d[clusterIndex] = streaming.resident.clasAddresses.d[clusterID + c];
    }
    
    subgroupClustersOffset = subgroupClustersOffsetNext;
  }
}

