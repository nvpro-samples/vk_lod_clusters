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

#include "shaderio_streaming.h"

#ifndef _SHADERIO_BUILDING_H_
#define _SHADERIO_BUILDING_H_

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else

#define INSTANCE_FRUSTUM_BIT 1
#define INSTANCE_VISIBLE_BIT 2

#endif

// The item descriptor used in the lod hierarchy traversal
// producer/consumer queue.
// It can can encode a lod hierarchy node, or a cluster group of an instance.
// must fit in 64-bit
struct TraversalInfo
{
  uint32_t instanceID;
  uint32_t packedNode;
};
#ifndef __cplusplus
TraversalInfo unpackTraversalInfo(uint64_t packed64) {
  u32vec2 data = unpack32(packed64);
  TraversalInfo info;
  info.instanceID = data.x;
  info.packedNode = data.y;
  return info;
}
uint64_t packTraversalInfo(TraversalInfo info)
{
  return pack64(u32vec2(info.instanceID,info.packedNode));
}
#endif

// A renderable cluster
// must fit in 64-bit, and can be overlayed with `TraversalInfo`
// thereore instanceID must come first.
struct ClusterInfo
{
  uint32_t instanceID;
  uint32_t clusterID;
};
BUFFER_REF_DECLARE_ARRAY(ClusterInfos_inout, ClusterInfo, , 8);

// Indirect build information to build a BLAS from an array of CLAS references
struct BlasBuildInfo
{
  // the number of CLAS that this BLAS references
  uint32_t clusterReferencesCount;
  // stride of array (typically 8 for 64-bit)
  uint32_t clusterReferencesStride;
  // start address of the array
  uint64_t clusterReferences;
};
BUFFER_REF_DECLARE_ARRAY(BlasBuildInfo_inout, BlasBuildInfo, , 16);

// Indirect build information for a TLAS instance
struct TlasInstance
{
  mat3x4    worldMatrix;
  uint32_t  instanceCustomIndex24_mask8;
  uint32_t  instanceShaderBindingTableRecordOffset24_flags8;
  uint64_t  blasReference;
};
BUFFER_REF_DECLARE_ARRAY(TlasInstances_inout, TlasInstance, , 16);

// The central structure that contains relevant information to
// perform the runtime lod hierchy traversal and building of 
// all relevant clusters to be rendered in the current frame.
// (not optimally packed for cache efficiency but readability)
struct SceneBuilding
{
  mat4  traversalViewMatrix;

  uint  numRenderInstances;
  uint  maxRenderClusters;
  uint  maxTraversalInfos;
  float errorOverDistanceThreshold;
  
  uint renderClusterCounter;
  int  traversalTaskCounter;
  uint traversalInfoReadCounter;
  uint traversalInfoWriteCounter;
  
  // result of traversal init & scratch for traversal run
  BUFFER_REF(uint64s_coh_volatile) traversalNodeInfos;
  // result of traversal run
  BUFFER_REF(ClusterInfos_inout) renderClusterInfos;
  
  // rasterization related
  //////////////////////////////////////////////////
  
  DrawMeshTasksIndirectCommandNV indirectDrawClusters;
  
  // ray tracing related
  //////////////////////////////////////////////////
  
  DispatchIndirectCommand indirectDispatchBlasInsertion;
  
  uint blasClasCounter;
  
  // instance states store culling/visibility related information
  BUFFER_REF(uint32s_inout) instanceStates;
  BUFFER_REF(TlasInstances_inout) tlasInstances;
  
  // per instance
  BUFFER_REF(BlasBuildInfo_inout) blasBuildInfos;
  BUFFER_REF(uint32s_inout) blasBuildSizes;
  // split into per-instance regions
  BUFFER_REF(uint64s_inout) blasClusterAddresses;
  uint64_t blasBuildData;
};



#ifdef __cplusplus
}
#endif
#endif // _SHADERIO_BUILDING_H_