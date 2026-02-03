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

#include "shaderio_streaming.h"

#ifndef _SHADERIO_BUILDING_H_
#define _SHADERIO_BUILDING_H_

#define TRAVERSAL_INVALID_LOD_LEVEL 0xFF

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else

#define INSTANCE_VISIBLE_BIT 1
#define INSTANCE_USES_MERGED_BIT 2

#define BLAS_BUILD_INDEX_LOWDETAIL (uint(~0))
#define BLAS_BUILD_INDEX_SHARE_BIT (uint(1 << 31))
#define BLAS_BUILD_INDEX_CACHE_BIT (uint(1 << 30))

// disabling is more for testing
#define TRAVERSAL_ALLOW_LOW_DETAIL_BLAS true

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
  mat3x4   worldMatrix;
  uint32_t instanceCustomIndex24_mask8;
  uint32_t instanceShaderBindingTableRecordOffset24_flags8;
  uint64_t blasReference;
};
BUFFER_REF_DECLARE_ARRAY(TlasInstances_inout, TlasInstance, , 16);

struct GeometryBuildHistogram
{
  // number of instances that start at this lod-level
  uint32_t lodLevelMinHistogram[SHADERIO_MAX_LOD_LEVELS];
  // number of instances that end at this lod-level
  uint32_t lodLevelMaxHistogram[SHADERIO_MAX_LOD_LEVELS];
  // a "somewhat" stable instance to use for sharing, that ends
  // at this lod-level
  uint32_t lodLevelMaxPackedInstance[SHADERIO_MAX_LOD_LEVELS];
};
BUFFER_REF_DECLARE_ARRAY(GeometryBuildHistogram_inout, GeometryBuildHistogram, , 16);

struct GeometryBuildInfo
{
  // blas caching
  uint32_t cachedBuildIndex;
  uint16_t cachedLevel;
  // blas sharing
  uint8_t  shareLevelMin;    // highest potential detail
  uint8_t  shareLevelMax;    // lowest potential detail
  uint32_t shareInstanceID;  // which instance to use for sharing (its lod range is expressed by the values above)
  // blas merging
  uint32_t mergedInstanceID;  // which instance to use for triggering the merged traversal
};
BUFFER_REF_DECLARE_ARRAY(GeometryBuildInfos_inout, GeometryBuildInfo, , 16);

struct InstanceBuildInfo
{
  // how many clusters this instance uses
  uint32_t clusterReferencesCount;

  // which blas to use
  // Can be BLAS_BUILD_INDEX_LOWDETAIL or have BLAS_BUILD_INDEX_SHARE_BIT set.
  // With BLAS_BUILD_INDEX_SHARE_BIT it encodes the instanceID that it shares the blas from.
  // Without any of the previous special cases it is the index into the array of the
  // dynamically built BLAS.
  uint32_t blasBuildIndex;

  uint8_t  lodLevelMin;          // highest potential detail
  uint8_t  lodLevelMax;          // lowest potential detail
  uint16_t geometryLodLevelMax;  // the highest lod level of the geometry
  uint32_t geometryID;
};
BUFFER_REF_DECLARE_ARRAY(InstanceBuildInfos_inout, InstanceBuildInfo, , 16);

// The central structure that contains relevant information to
// perform the runtime lod hierchy traversal and building of
// all relevant clusters to be rendered in the current frame.
// (not optimally packed for cache efficiency but readability)
struct SceneBuilding
{
  mat4 traversalViewMatrix;
  mat4 cullViewProjMatrix;
  mat4 cullViewProjMatrixLast;

  // for two pass culling
  uint pass;
  uint frameIndex;

  uint numGeometries;
  uint numRenderInstances;
  uint maxRenderClusters;
  uint maxTraversalInfos;

  float errorOverDistanceThreshold;
  float culledErrorScale;
  float swRasterThreshold;

  uint sharingEnabledLevels;
  uint sharingTolerantLevels;
  uint sharingPushCulled;

  uint renderClusterCounter;
  uint renderClusterCounterSW;

  int  traversalTaskCounter;
  uint traversalInfoReadCounter;
  uint traversalInfoWriteCounter;
  // only used for USE_SEPARATE_GROUPS
  uint traversalGroupCounter;

  // result of traversal init & scratch for traversal run
  // array size is [maxTraversalInfos]
  BUFFER_REF(uint64s_coh_volatile) traversalNodeInfos;
  // only used for USE_SEPARATE_GROUPS
  // array size is [maxTraversalInfos]
  BUFFER_REF(uint64s_coh_volatile) traversalGroupInfos;

  // result of traversal run or separate_groups
  // array size is [maxRenderClusters]
  BUFFER_REF(ClusterInfos_inout) renderClusterInfos;

  // only used for USE_SEPARATE_GROUPS
  DispatchIndirectCommand indirectDispatchGroups;

  // rasterization related
  //////////////////////////////////////////////////

  // hw rasterization via nv
  DrawMeshTasksIndirectCommandNV indirectDrawClustersNV;
  DrawMeshTasksIndirectCommandNV indirectDrawClusterBoxesNV;

  // hw rasterization via ext
  DrawMeshTasksIndirectCommandEXT indirectDrawClustersEXT;
  DrawMeshTasksIndirectCommandEXT indirectDrawClusterBoxesEXT;
  uint                            numRenderedClusters;

  // sw rasterization
  DispatchIndirectCommand indirectDrawClustersSW;
  uint                    numRenderedClustersSW;

  // clusters classified for sw-rasterization.
  // We simply have two lists, we could use a single list
  // and append front or back to conserve memory, however
  // for simplicity two are used.
  BUFFER_REF(ClusterInfos_inout) renderClusterInfosSW;

  // ray tracing related
  //////////////////////////////////////////////////

  DispatchIndirectCommand indirectDispatchBlasInsertion;

  // Computed dynamically, total number of CLAS that are referenced
  // across all BLAS built in a frame.
  uint blasClasCounter;

  // Computed dynamically and contains the number of blas that are built in a frame.
  // It is read indirectly by `vkCmdBuildClusterAccelerationStructureIndirectNV`
  uint blasBuildCounter;

  // per scene instance
  // ------------

  // culling/visibility related information
  BUFFER_REF(uint8s_inout) instanceVisibility;

  // instance sorting
  BUFFER_REF(uint32s_inout) instanceSortValues;
  BUFFER_REF(uint32s_inout) instanceSortKeys;

  // instance blas handling
  BUFFER_REF(InstanceBuildInfos_inout) instanceBuildInfos;
  BUFFER_REF(TlasInstances_inout) tlasInstances;

  // per blas build
  // --------------
  // (arrays are sized for worst-case to be per-instance)

  // configured in `blas_setup_insertion.comp.glsl`
  // and input to `vkCmdBuildClusterAccelerationStructureIndirectNV`
  BUFFER_REF(BlasBuildInfo_inout) blasBuildInfos;

  // result of building operations
  BUFFER_REF(uint32s_inout) blasBuildSizes;
  BUFFER_REF(uint64s_inout) blasBuildAddresses;

  // split into per-blas regions during
  // `blas_setup_insertion.comp.glsl`
  // array size is [maxRenderClusters]
  BUFFER_REF(uint64s_inout) blasClusterAddresses;

  // per scene geometry
  // ------------
  BUFFER_REF(GeometryBuildInfos_inout) geometryBuildInfos;
  BUFFER_REF(GeometryBuildHistogram_inout) geometryHistograms;

  // for USE_BLAS_CACHING
  BUFFER_REF(uint64s_inout) cachedBlasClusterAddressesSrc;
  BUFFER_REF(uint64s_inout) cachedBlasClusterAddressesDst;

  uint32_t cachedBlasCopyCounter;
};


#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_BUILDING_H_