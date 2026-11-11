/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resources.hpp"

namespace lodclusters {

// Storage and build of a partitioned top level acceleration structure (PTLAS)
// using `VK_NV_partitioned_acceleration_structure`.
//
// In contrast to a regular TLAS all build inputs live on the device: the operations
// to perform and their arguments are filled in by compute shaders, this class only
// provides the memory and issues the indirect build.
// The buffers holding operations and arguments are owned by the renderer, they are
// part of its `shaderio::SceneBuilding`. So is the scratch, which it shares with other builds.

class PartitionedTlas
{
public:
  struct Config
  {
    // total number of instances, their `instanceIndex` must be < this
    uint32_t instanceCount = 0;
    // number of spatial partitions, `partitionIndex` must be < this
    uint32_t partitionCount = 0;
    // worst-case occupancy of a single partition
    uint32_t maxInstancesPerPartition = 0;
    // instances within PTLAS_PARTITION_GLOBAL, unused for static scenes
    uint32_t maxInstancesInGlobalPartition = 0;

    VkBuildAccelerationStructureFlagsKHR buildFlags = 0;
  };

  VkResult init(Resources& res, const Config& config);
  void     deinit(Resources& res);

  // `operations` is an array of `shaderio::PtlasOperation`, `operationCount` a single uint32_t,
  // both are read on the device. `update` sources the previous state of this PTLAS, otherwise
  // the operations must describe all instances. `scratch` must hold `getScratchSize()` bytes.
  void cmdBuild(VkCommandBuffer cmd, VkDeviceAddress operations, VkDeviceAddress operationCount, VkDeviceAddress scratch, bool update);

  // the shaders convert this address into an `accelerationStructureEXT`,
  // a PTLAS has no acceleration structure object to bind
  VkDeviceAddress getAddress() const { return m_accelerationStructure.address; }

  VkDeviceSize getAccelerationStructureSize() const { return m_accelerationStructure.bufferSize; }
  VkDeviceSize getScratchSize() const { return m_scratchSize; }

  // the PTLAS and the build scratch must be 256 byte aligned (VUID-10542, -10545)
  static constexpr VkDeviceSize ADDRESS_ALIGNMENT = 256;

  // properties of the device, `maxPartitionCount` limits `Config::partitionCount`
  static VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV getProperties(VkPhysicalDevice physicalDevice);

private:
  VkPartitionedAccelerationStructureInstancesInputNV m_input{VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV};
  VkPartitionedAccelerationStructureFlagsNV m_flags{VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV};

  nvvk::Buffer m_accelerationStructure;
  VkDeviceSize m_scratchSize = 0;
};

// the shaderio structs are filled in by the device, they must match the extension's
static_assert(sizeof(shaderio::PtlasInstanceWrite) == sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV));
static_assert(offsetof(shaderio::PtlasInstanceWrite, bboxLo)
              == offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, explicitAABB));
static_assert(offsetof(shaderio::PtlasInstanceWrite, instanceCustomIndex)
              == offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, instanceID));
static_assert(offsetof(shaderio::PtlasInstanceWrite, partitionIndex)
              == offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, partitionIndex));
static_assert(offsetof(shaderio::PtlasInstanceWrite, blasReference)
              == offsetof(VkPartitionedAccelerationStructureWriteInstanceDataNV, accelerationStructure));

static_assert(sizeof(shaderio::PtlasInstanceWrite) == PTLAS_INSTANCE_WRITE_SIZE);
static_assert(sizeof(shaderio::PtlasInstanceUpdate) == PTLAS_INSTANCE_UPDATE_SIZE);

static_assert(sizeof(shaderio::PtlasInstanceUpdate) == sizeof(VkPartitionedAccelerationStructureUpdateInstanceDataNV));
static_assert(offsetof(shaderio::PtlasInstanceUpdate, blasReference)
              == offsetof(VkPartitionedAccelerationStructureUpdateInstanceDataNV, accelerationStructure));

static_assert(sizeof(shaderio::PtlasOperation) == sizeof(VkBuildPartitionedAccelerationStructureIndirectCommandNV));
static_assert(offsetof(shaderio::PtlasOperation, argAddress)
              == offsetof(VkBuildPartitionedAccelerationStructureIndirectCommandNV, argData));

static_assert(PTLAS_OP_WRITE_INSTANCE == VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV);
static_assert(PTLAS_OP_UPDATE_INSTANCE == VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_UPDATE_INSTANCE_NV);
static_assert(PTLAS_PARTITION_GLOBAL == VK_PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV);
static_assert(PtlasInstanceFlag_EXPLICIT_BBOX_BIT == VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_ENABLE_EXPLICIT_BOUNDING_BOX_NV);
static_assert(PtlasInstanceFlag_FLIP_FACING_BIT == VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FLIP_FACING_BIT_NV);

}  // namespace lodclusters
