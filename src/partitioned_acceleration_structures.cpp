/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <volk.h>
#include <nvutils/logger.hpp>

#include "partitioned_acceleration_structures.hpp"

namespace lodclusters {

VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV PartitionedTlas::getProperties(VkPhysicalDevice physicalDevice)
{
  VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV ptlasProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES_NV};
  VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &ptlasProperties};
  vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
  return ptlasProperties;
}

VkResult PartitionedTlas::init(Resources& res, const Config& config)
{
  m_input.flags                             = config.buildFlags;
  m_input.instanceCount                     = config.instanceCount;
  m_input.partitionCount                    = config.partitionCount;
  m_input.maxInstancePerPartitionCount      = config.maxInstancesPerPartition;
  m_input.maxInstanceInGlobalPartitionCount = config.maxInstancesInGlobalPartition;
  m_input.pNext                             = &m_flags;

  VkAccelerationStructureBuildSizesInfoKHR sizesInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetPartitionedAccelerationStructuresBuildSizesNV(res.m_device, &m_input, &sizesInfo);

  VkResult result = res.createBuffer(m_accelerationStructure, sizesInfo.accelerationStructureSize,
                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, ADDRESS_ALIGNMENT);
  if(result != VK_SUCCESS)
  {
    return result;
  }
  NVVK_DBG_NAME(m_accelerationStructure.buffer);

  m_scratchSize = sizesInfo.buildScratchSize;

  LOGI("raytracer: PTLAS size %d KiB, build scratchsize %d KiB\n",
       uint32_t((sizesInfo.accelerationStructureSize + 1023) / 1024), uint32_t((sizesInfo.buildScratchSize + 1023) / 1024));

  return VK_SUCCESS;
}

void PartitionedTlas::deinit(Resources& res)
{
  res.m_allocator.destroyBuffer(m_accelerationStructure);
  *this = {};
}

void PartitionedTlas::cmdBuild(VkCommandBuffer cmd, VkDeviceAddress operations, VkDeviceAddress operationCount, VkDeviceAddress scratch, bool update)
{
  VkBuildPartitionedAccelerationStructureInfoNV buildInfo{VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV};
  buildInfo.input                        = m_input;
  buildInfo.srcAccelerationStructureData = update ? m_accelerationStructure.address : 0;
  buildInfo.dstAccelerationStructureData = m_accelerationStructure.address;
  buildInfo.scratchData                  = scratch;
  buildInfo.srcInfos                     = operations;
  buildInfo.srcInfosCount                = operationCount;

  vkCmdBuildPartitionedAccelerationStructuresNV(cmd, &buildInfo);
}

}  // namespace lodclusters
