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

#pragma once

#include <vulkan/vulkan_core.h>

//////////////////////////////////////////////////////////////////////////
//
// Two extensions, which can be used independently, make the "RTX Mega Geometry" feature:
//
// # VK_NV_cluster_acceleration_structure
//
// Clusters contain content like triangles, and are then referenced within
// one or more bottom-level acceleration structures. Referencing allows similar
// memory saving like instances, but is without transforms. The clusters are also called "CLAS".
//
// Cluster templates allow quicker building of clusters for the purpose of
// animation or "micro-instancing" topology.
//
// # VK_NV_partitioned_acceleration_structure
//
// Partitions divide a fixed pool with a maximum size of number of instances across a top-level
// acceleration structure (AS). The feature is also referred to as "PTLAS".
//
// # Common
//
// Both new extensions are "multi indirect", however with slightly different designs.
// Cluster builds are one type of operation per single commandbuffer command, following
// the traditional indirect approach.
//
// Partition builds/updates use two level-indirection, meaning multiple operation types
// can be executed per single commandbuffer command, and the types are also sourced
// from GPU


#ifndef VK_NV_cluster_acceleration_structure
#define VK_NV_cluster_acceleration_structure 1
#define VK_NV_CLUSTER_ACCELERATION_STRUCTURE_SPEC_VERSION 2
#define VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME "VK_NV_cluster_acceleration_structure"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV ((VkStructureType)1000569000)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV ((VkStructureType)1000569001)
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV ((VkStructureType)1000569002)
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV ((VkStructureType)1000569003)
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV ((VkStructureType)1000569004)
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV ((VkStructureType)1000569005)
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV ((VkStructureType)1000569006)
#define VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV                           \
  ((VkStructureType)1000569007)
#define VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_FLAGS_NV ((VkStructureType)1000569008)
#define VK_OPACITY_MICROMAP_SPECIAL_INDEX_CLUSTER_GEOMETRY_DISABLE_OPACITY_MICROMAP_NV                                 \
  ((VkOpacityMicromapSpecialIndexEXT) - 5)

typedef struct VkPhysicalDeviceClusterAccelerationStructureFeaturesNV
{
  VkStructureType sType;
  void*           pNext;
  VkBool32        clusterAccelerationStructures;
} VkPhysicalDeviceClusterAccelerationStructureFeaturesNV;

typedef struct VkPhysicalDeviceClusterAccelerationStructurePropertiesNV
{
  VkStructureType sType;
  void*           pNext;
  uint32_t        maxVerticesPerCluster;
  uint32_t        maxTrianglesPerCluster;
  uint32_t        clusterScratchByteAlignment;
  uint32_t        clusterByteAlignment;
  uint32_t        clusterTemplateByteAlignment;
  uint32_t        clusterBottomLevelByteAlignment;
  uint32_t        clusterTemplateBoundsByteAlignment;
  uint32_t        maxClusterGeometryIndex;
} VkPhysicalDeviceClusterAccelerationStructurePropertiesNV;

typedef struct VkClusterAccelerationStructureClustersBottomLevelInputNV
{
  VkStructureType sType;
  void*           pNext;
  uint32_t        maxTotalClusterCount;
  uint32_t        maxClusterCountPerAccelerationStructure;
} VkClusterAccelerationStructureClustersBottomLevelInputNV;

typedef struct VkClusterAccelerationStructureTriangleClusterInputNV
{
  VkStructureType sType;
  void*           pNext;
  VkFormat        vertexFormat;
  uint32_t        maxGeometryIndexValue;
  uint32_t        maxClusterUniqueGeometryCount;
  uint32_t        maxClusterTriangleCount;
  uint32_t        maxClusterVertexCount;
  uint32_t        maxTotalTriangleCount;
  uint32_t        maxTotalVertexCount;
  uint32_t        minPositionTruncateBitCount;
} VkClusterAccelerationStructureTriangleClusterInputNV;

typedef enum VkClusterAccelerationStructureTypeNV
{
  VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_CLUSTERS_BOTTOM_LEVEL_NV     = 0,
  VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV          = 1,
  VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_TEMPLATE_NV = 2,
  VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_NV                  = 0x7FFFFFFF
} VkClusterAccelerationStructureTypeNV;

typedef struct VkClusterAccelerationStructureMoveObjectsInputNV
{
  VkStructureType                      sType;
  void*                                pNext;
  VkClusterAccelerationStructureTypeNV type;
  VkBool32                             noMoveOverlap;
  VkDeviceSize                         maxMovedBytes;
} VkClusterAccelerationStructureMoveObjectsInputNV;

typedef union VkClusterAccelerationStructureOpInputNV
{
  VkClusterAccelerationStructureClustersBottomLevelInputNV* pClustersBottomLevel;
  VkClusterAccelerationStructureTriangleClusterInputNV*     pTriangleClusters;
  VkClusterAccelerationStructureMoveObjectsInputNV*         pMoveObjects;
} VkClusterAccelerationStructureOpInputNV;

typedef enum VkClusterAccelerationStructureOpTypeNV
{
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV                    = 0,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV     = 1,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV          = 2,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV = 3,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV    = 4,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MAX_ENUM_NV                        = 0x7FFFFFFF
} VkClusterAccelerationStructureOpTypeNV;

typedef enum VkClusterAccelerationStructureOpModeNV
{
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV = 0,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV = 1,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV         = 2,
  VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_MAX_ENUM_NV              = 0x7FFFFFFF
} VkClusterAccelerationStructureOpModeNV;

typedef struct VkClusterAccelerationStructureInputInfoNV
{
  VkStructureType                         sType;
  void*                                   pNext;
  uint32_t                                maxAccelerationStructureCount;
  VkBuildAccelerationStructureFlagsKHR    flags;
  VkClusterAccelerationStructureOpTypeNV  opType;
  VkClusterAccelerationStructureOpModeNV  opMode;
  VkClusterAccelerationStructureOpInputNV opInput;
} VkClusterAccelerationStructureInputInfoNV;

typedef VkFlags VkClusterAccelerationStructureAddressResolutionFlagsNV;

typedef struct VkClusterAccelerationStructureCommandsInfoNV
{
  VkStructureType                                        sType;
  void*                                                  pNext;
  VkClusterAccelerationStructureInputInfoNV              input;
  VkDeviceAddress                                        dstImplicitData;
  VkDeviceAddress                                        scratchData;
  VkStridedDeviceAddressRegionKHR                        dstAddressesArray;
  VkStridedDeviceAddressRegionKHR                        dstSizesArray;
  VkStridedDeviceAddressRegionKHR                        srcInfosArray;
  VkDeviceAddress                                        srcInfosCount;
  VkClusterAccelerationStructureAddressResolutionFlagsNV addressResolutionFlags;
} VkClusterAccelerationStructureCommandsInfoNV;

typedef struct VkStridedDeviceAddressNV
{
  VkDeviceAddress startAddress;
  VkDeviceSize    strideInBytes;
} VkStridedDeviceAddressNV;

typedef struct VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV
{
  VkStructureType sType;
  void*           pNext;
  VkBool32        allowClusterAccelerationStructures;
} VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV;

typedef struct VkClusterAccelerationStructureMoveObjectsInfoNV
{
  VkDeviceAddress srcAccelerationStructure;
} VkClusterAccelerationStructureMoveObjectsInfoNV;

typedef struct VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV
{
  uint32_t        clusterReferencesCount;
  uint32_t        clusterReferencesStride;
  VkDeviceAddress clusterReferences;
} VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV;

typedef struct VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV
{
  uint32_t geometryIndex : 24;
  uint32_t reserved : 5;
  uint32_t geometryFlags : 3;
} VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV;

typedef VkFlags VkClusterAccelerationStructureClusterFlagsNV;

typedef struct VkClusterAccelerationStructureBuildTriangleClusterInfoNV
{
  uint32_t                                                      clusterID;
  VkClusterAccelerationStructureClusterFlagsNV                  clusterFlags;
  uint32_t                                                      triangleCount : 9;
  uint32_t                                                      vertexCount : 9;
  uint32_t                                                      positionTruncateBitCount : 6;
  uint32_t                                                      indexType : 4;
  uint32_t                                                      opacityMicromapIndexType : 4;
  VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV baseGeometryIndexAndGeometryFlags;
  uint16_t                                                      indexBufferStride;
  uint16_t                                                      vertexBufferStride;
  uint16_t                                                      geometryIndexAndFlagsBufferStride;
  uint16_t                                                      opacityMicromapIndexBufferStride;
  VkDeviceAddress                                               indexBuffer;
  VkDeviceAddress                                               vertexBuffer;
  VkDeviceAddress                                               geometryIndexAndFlagsBuffer;
  VkDeviceAddress                                               opacityMicromapArray;
  VkDeviceAddress                                               opacityMicromapIndexBuffer;
} VkClusterAccelerationStructureBuildTriangleClusterInfoNV;

typedef struct VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV
{
  uint32_t                                                      clusterID;
  VkClusterAccelerationStructureClusterFlagsNV                  clusterFlags;
  uint32_t                                                      triangleCount : 9;
  uint32_t                                                      vertexCount : 9;
  uint32_t                                                      positionTruncateBitCount : 6;
  uint32_t                                                      indexType : 4;
  uint32_t                                                      opacityMicromapIndexType : 4;
  VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV baseGeometryIndexAndGeometryFlags;
  uint16_t                                                      indexBufferStride;
  uint16_t                                                      vertexBufferStride;
  uint16_t                                                      geometryIndexAndFlagsBufferStride;
  uint16_t                                                      opacityMicromapIndexBufferStride;
  VkDeviceAddress                                               indexBuffer;
  VkDeviceAddress                                               vertexBuffer;
  VkDeviceAddress                                               geometryIndexAndFlagsBuffer;
  VkDeviceAddress                                               opacityMicromapArray;
  VkDeviceAddress                                               opacityMicromapIndexBuffer;
  VkDeviceAddress                                               instantiationBoundingBoxLimit;
} VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV;

typedef enum VkClusterAccelerationStructureClusterFlagBitsNV
{
  VK_CLUSTER_ACCELERATION_STRUCTURE_CLUSTER_ALLOW_DISABLE_OPACITY_MICROMAPS_NV = 0x00000001,
  VK_CLUSTER_ACCELERATION_STRUCTURE_CLUSTER_FLAG_BITS_MAX_ENUM_NV              = 0x7FFFFFFF
} VkClusterAccelerationStructureClusterFlagBitsNV;

typedef VkFlags VkClusterAccelerationStructureGeometryFlagsNV;

typedef enum VkClusterAccelerationStructureGeometryFlagBitsNV
{
  VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_CULL_DISABLE_BIT_NV                   = 0x00000001,
  VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_NO_DUPLICATE_ANYHIT_INVOCATION_BIT_NV = 0x00000002,
  VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV                         = 0x00000004,
  VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_FLAG_BITS_MAX_ENUM_NV                 = 0x7FFFFFFF
} VkClusterAccelerationStructureGeometryFlagBitsNV;

typedef enum VkClusterAccelerationStructureAddressResolutionFlagBitsNV
{
  VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_IMPLICIT_DATA_BIT_NV = 0x00000001,
  VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SCRATCH_DATA_BIT_NV      = 0x00000002,
  VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_ADDRESS_ARRAY_BIT_NV = 0x00000004,
  VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_SIZES_ARRAY_BIT_NV   = 0x00000008,
  VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SRC_INFOS_ARRAY_BIT_NV   = 0x00000010,
  VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SRC_INFOS_COUNT_BIT_NV   = 0x00000020,
  VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_FLAG_BITS_MAX_ENUM_NV               = 0x7FFFFFFF
} VkClusterAccelerationStructureAddressResolutionFlagBitsNV;

typedef struct VkClusterAccelerationStructureInstantiateClusterInfoNV
{
  uint32_t                 clusterIdOffset;
  uint32_t                 geometryIndexOffset : 24;
  uint32_t                 reserved : 8;
  VkDeviceAddress          clusterTemplateAddress;
  VkStridedDeviceAddressNV vertexBuffer;
} VkClusterAccelerationStructureInstantiateClusterInfoNV;

typedef enum VkClusterAccelerationStructureIndexFormatNV
{
  VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV     = 0x00000001,
  VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV    = 0x00000002,
  VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV    = 0x00000004,
  VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_MAX_ENUM_NV = 0x7FFFFFFF
} VkClusterAccelerationStructureIndexFormatNV;

typedef void(VKAPI_PTR* PFN_vkGetClusterAccelerationStructureBuildSizesNV)(VkDevice device,
                                                                           const VkClusterAccelerationStructureInputInfoNV* pInfo,
                                                                           VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo);
typedef void(VKAPI_PTR* PFN_vkCmdBuildClusterAccelerationStructureIndirectNV)(VkCommandBuffer commandBuffer,
                                                                              const VkClusterAccelerationStructureCommandsInfoNV* pCommandInfos);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR void VKAPI_CALL vkGetClusterAccelerationStructureBuildSizesNV(VkDevice device,
                                                                         VkClusterAccelerationStructureInputInfoNV const* pInfo,
                                                                         VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo);

VKAPI_ATTR void VKAPI_CALL vkCmdBuildClusterAccelerationStructureIndirectNV(VkCommandBuffer commandBuffer,
                                                                            VkClusterAccelerationStructureCommandsInfoNV const* pCommandInfos);
#endif
#endif

VkBool32 load_VK_NV_cluster_accleration_structure(VkInstance instance, VkDevice device);
