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

#include "scene_streaming_utils.hpp"

namespace lodclusters {

//////////////////////////////////////////////////////////////////////////
//
// StreamingRequests

void StreamingRequests::init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment)
{
  m_shaderData            = {};
  m_shaderData.maxLoads   = config.maxPerFrameLoadRequests;
  m_shaderData.maxUnloads = config.maxPerFrameUnloadRequests;

  // some values are aligned up for easier gpu kernel access

  BufferRanges ranges = {};
  m_shaderData.loadGeometryGroups =
      ranges.append(sizeof(GeometryGroup) * nvh::align_up(config.maxPerFrameLoadRequests, groupCountAlignment), 8);
  m_shaderData.unloadGeometryGroups =
      ranges.append(sizeof(GeometryGroup) * nvh::align_up(config.maxPerFrameUnloadRequests, groupCountAlignment), 8);
  m_requestSize = ranges.getSize();

  // must come after the others
  m_shaderDataOffset = m_requestSize * STREAMING_MAX_ACTIVE_TASKS;

  std::vector<uint32_t> sharingQueueFamilies;
  if(config.useAsyncTransfer)
  {
    sharingQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);
    sharingQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);
  }

  m_requestBuffer = res.createBuffer((m_requestSize + sizeof(shaderio::StreamingRequest)) * STREAMING_MAX_ACTIVE_TASKS,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &sharingQueueFamilies);

  m_requestHostBuffer = res.createBuffer(m_requestBuffer.info.range, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  for(uint32_t c = 0; c < STREAMING_MAX_ACTIVE_TASKS; c++)
  {
    TaskInfo& task = m_taskInfos[c];
    task           = {};

    task.shaderData = reinterpret_cast<const shaderio::StreamingRequest*>(
        uint64_t(m_requestHostBuffer.mapping) + m_shaderDataOffset + sizeof(shaderio::StreamingRequest) * c);
    task.loadGeometryGroups = reinterpret_cast<const GeometryGroup*>(
        uint64_t(m_requestHostBuffer.mapping) + m_requestSize * c + m_shaderData.loadGeometryGroups);
    task.unloadGeometryGroups = reinterpret_cast<const GeometryGroup*>(
        uint64_t(m_requestHostBuffer.mapping) + m_requestSize * c + m_shaderData.unloadGeometryGroups);
  }
}

void StreamingRequests::deinit(Resources& res)
{
  res.destroy(m_requestBuffer);
  res.destroy(m_requestHostBuffer);
}

size_t StreamingRequests::getOperationsSize() const
{
  return m_requestBuffer.info.range;
}

void StreamingRequests::applyTask(shaderio::StreamingRequest& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
  shaderData = m_shaderData;
  shaderData.loadGeometryGroups += m_requestBuffer.address + m_requestSize * taskIndex;
  shaderData.unloadGeometryGroups += m_requestBuffer.address + m_requestSize * taskIndex;
  shaderData.taskIndex = taskIndex;
  // special address value that allows us to ensure that a non-resident geometry group
  // isn't requested multiple times in the same frame.
  shaderData.frameIndex = STREAMING_INVALID_ADDRESS_START + frameIndex;
}

void StreamingRequests::cmdRunTask(VkCommandBuffer cmd, const shaderio::StreamingRequest& shaderData, VkBuffer buffer, size_t bufferOffset)
{
  uint32_t taskIndex = shaderData.taskIndex;

  // copy the newly requested indices to host
  VkBufferCopy region;
  region.dstOffset = m_requestSize * taskIndex;
  region.srcOffset = m_requestSize * taskIndex;
  region.size      = m_requestSize;
  vkCmdCopyBuffer(cmd, m_requestBuffer.buffer, m_requestHostBuffer.buffer, 1, &region);

  // copy the shaderio, actually we only care for the counters of the request
  // but grabbing everything is useful for pointer comparisons
  region.dstOffset = m_shaderDataOffset + (sizeof(shaderio::StreamingRequest) * taskIndex);
  region.srcOffset = bufferOffset;
  region.size      = sizeof(shaderio::StreamingRequest);
  vkCmdCopyBuffer(cmd, buffer, m_requestHostBuffer.buffer, 1, &region);
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingResident

void StreamingResident::init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment)
{
  m_groupAllocator.init(config.maxGroups);
  m_clusterAllocator.init(config.maxClusters);

  // some values are aligned up for easier gpu kernel access

  m_maxClusters  = nvh::align_up(config.maxClusters, clusterCountAlignment);
  m_maxGroups    = nvh::align_up(config.maxGroups, groupCountAlignment);
  m_maxClasBytes = 0;

  m_lowDetailGroupsCount   = 0;
  m_lowDetailClustersCount = 0;

  m_activeGroupsCount   = 0;
  m_activeClustersCount = 0;

  m_groupIndicesUpdateRange    = {};
  m_groups                     = {};
  m_activeGroupIndices         = {};
  m_mapGeometryGroup2Residency = {};

  m_mapGeometryGroup2Residency.reserve(m_maxGroups);
  m_groups.resize(m_maxGroups);
  m_activeGroupIndices.resize(m_maxGroups);

  BufferRanges ranges      = {};
  m_residentGroupsOffset   = ranges.append(sizeof(shaderio::StreamingGroup) * m_maxGroups, 16);
  m_residentClustersOffset = ranges.append(sizeof(shaderio::uint64_t) * m_maxClusters, 8);
  m_residentActiveOffset   = ranges.append(sizeof(shaderio::uint32_t) * m_maxGroups, 4);
  m_residentActiveUpdateOffset = ranges.append(sizeof(shaderio::uint32_t) * m_maxGroups * STREAMING_MAX_ACTIVE_TASKS, 4);

  std::vector<uint32_t> sharingQueueFamilies;
  if(config.useAsyncTransfer)
  {
    sharingQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);
    sharingQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);
  }

  m_residentBuffer = res.createBuffer(ranges.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &sharingQueueFamilies);

  res.createBufferTyped(m_residentActiveHostBuffer, (m_maxGroups)*STREAMING_MAX_ACTIVE_TASKS, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  m_shaderData              = {};
  m_shaderData.groups       = m_residentBuffer.address + m_residentGroupsOffset;
  m_shaderData.clusters     = m_residentBuffer.address + m_residentClustersOffset;
  m_shaderData.activeGroups = m_residentBuffer.address + m_residentActiveOffset;
}

void StreamingResident::deinit(Resources& res)
{
  res.destroy(m_residentBuffer);
  res.destroy(m_residentActiveHostBuffer);
  deinitClas(res);

  *this = {};
}

size_t StreamingResident::getOperationsSize() const
{
  return m_residentBuffer.info.range;
}

const StreamingResident::Group* StreamingResident::initClas(Resources&                   res,
                                                            const StreamingConfig&       config,
                                                            shaderio::StreamingResident& shaderData,
                                                            uint32_t&                    loGroupsCount,
                                                            uint32_t&                    loClustersCount)
{
  m_maxClasBytes = config.maxClasMegaBytes * 1024 * 1024;

  BufferRanges ranges                    = {};
  m_shaderData.clasAddresses             = ranges.append(sizeof(shaderio::uint64_t) * m_maxClusters, 8);
  m_shaderData.clasSizes                 = ranges.append(sizeof(shaderio::uint32_t) * m_maxClusters, 4);
  m_shaderData.clasCompactionUsedSize    = ranges.append(sizeof(shaderio::uint64_t), 8);
  m_shaderData.clasAllocatedMaxSizedLeft = ranges.append(sizeof(shaderio::uint32_t), 4);
  if(config.usePersistentClasAllocator)
  {
    m_shaderData.groupClasSizes = ranges.append(sizeof(glm::uvec2) * m_maxGroups, 8);
  }

  // one buffer for organization
  m_clasManageBuffer = res.createBuffer(ranges.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  m_shaderData.clasAddresses += m_clasManageBuffer.address;
  m_shaderData.clasSizes += m_clasManageBuffer.address;
  m_shaderData.clasCompactionUsedSize += m_clasManageBuffer.address;
  m_shaderData.clasAllocatedMaxSizedLeft += m_clasManageBuffer.address;
  if(config.usePersistentClasAllocator)
  {
    m_shaderData.groupClasSizes += m_clasManageBuffer.address;
  }

  // one buffer for actual storage, allow > 4 GB
  m_clasDataBuffer             = res.createLargeBuffer(m_maxClasBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                           | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  m_shaderData.clasBaseAddress = m_clasDataBuffer.address;
  m_shaderData.clasMaxSize     = m_maxClasBytes;

  shaderData = m_shaderData;

  loGroupsCount   = m_lowDetailGroupsCount;
  loClustersCount = m_lowDetailClustersCount;

  return m_groups.data();
}

size_t StreamingResident::getClasOperationsSize() const
{
  return m_clasManageBuffer.info.range;
}

void StreamingResident::getStats(StreamingStats& stats) const
{
  stats.residentGroups     = m_activeGroupsCount;
  stats.residentClusters   = m_activeClustersCount;
  stats.persistentGroups   = m_lowDetailGroupsCount;
  stats.persistentClusters = m_lowDetailClustersCount;
}

void StreamingResident::deinitClas(Resources& res)
{
  res.destroy(m_clasManageBuffer);
  res.destroy(m_clasDataBuffer);

  m_shaderData.clasBaseAddress           = 0;
  m_shaderData.clasAddresses             = 0;
  m_shaderData.clasSizes                 = 0;
  m_shaderData.clasCompactionUsedSize    = 0;
  m_shaderData.clasAllocatedMaxSizedLeft = 0;
  m_shaderData.groupClasSizes            = 0;
  m_shaderData.clasMaxSize               = 0;
}

void StreamingResident::reset(shaderio::StreamingResident& shaderData)
{
  for(uint32_t activeGroup = m_lowDetailGroupsCount; activeGroup < m_activeGroupsCount; activeGroup++)
  {
    Group& group = m_groups[m_activeGroupIndices[activeGroup]];

    m_mapGeometryGroup2Residency.erase(group.geometryGroup.key);
    m_groupAllocator.subFree(group.groupResidentID, 1);
    m_clusterAllocator.subFree(group.clusterResidentID, group.clusterCount);
  }

  m_activeGroupsCount   = m_lowDetailGroupsCount;
  m_activeClustersCount = m_lowDetailClustersCount;

  m_groupIndicesUpdateRange = {};

  // adjust active list in shaders to skip persistent
  m_shaderData.activeGroupsCount   = m_activeGroupsCount - m_lowDetailGroupsCount;
  m_shaderData.activeClustersCount = m_activeClustersCount - m_lowDetailClustersCount;
  m_shaderData.activeGroups = m_residentBuffer.address + m_residentActiveOffset + sizeof(uint32_t) * m_lowDetailGroupsCount;

  shaderData = m_shaderData;
}


void StreamingResident::uploadInitialState(Resources::BatchedUploader& uploader, shaderio::StreamingResident& shaderData)
{
  // all groups and clusters added so far are part of the persistent low detail state

  m_lowDetailGroupsCount   = m_activeGroupsCount;
  m_lowDetailClustersCount = m_activeClustersCount;

  // for debugging (see STREAMING_DEBUG_ADDRESSES) set this to m_maxGroups, otherwise m_loGroupsCount
  uint32_t updatedActiveGroups = m_lowDetailGroupsCount;
#if STREAMING_DEBUG_ADDRESSES
  updatedActiveGroups = m_maxGroups;
#endif

  shaderio::StreamingGroup* shaderGroups =
      uploader.uploadBuffer(m_residentBuffer, m_residentGroupsOffset,
                            sizeof(shaderio::StreamingGroup) * updatedActiveGroups, (shaderio::StreamingGroup*)nullptr);

  uint64_t* shaderClusters = uploader.uploadBuffer(m_residentBuffer, m_residentClustersOffset,
                                                   sizeof(shaderio::uint64_t) * m_activeClustersCount,
                                                   (uint64_t*)nullptr, Resources::DONT_FLUSH);

  for(uint32_t g = 0; g < m_lowDetailGroupsCount; g++)
  {
    const Group& group = m_groups[g];
    assert(group.groupResidentID == g);

    shaderio::StreamingGroup& shaderGroup = shaderGroups[g];
    shaderGroup.age                       = -12345678;
    shaderGroup.clusterCount              = group.clusterCount;
    shaderGroup.group                     = group.deviceAddress;
    for(uint32_t c = 0; c < group.clusterCount; c++)
    {
      shaderClusters[group.clusterResidentID + c] = group.deviceAddress + sizeof(shaderio::Group) + sizeof(shaderio::Cluster) * c;
    }
  }
#if STREAMING_DEBUG_ADDRESSES
  // for debugging purposes pre-fill with invalid
  for(uint32_t g = m_lowDetailGroupsCount; g < updatedActiveGroups; g++)
  {
    shaderGroups[g].group = STREAMING_INVALID_ADDRESS_START;
  }
#endif

  // adjust active list in shaders to skip persistent
  m_shaderData.activeGroupsCount   = m_activeGroupsCount - m_lowDetailGroupsCount;
  m_shaderData.activeClustersCount = m_activeClustersCount - m_lowDetailClustersCount;
  m_shaderData.activeGroups = m_residentBuffer.address + m_residentActiveOffset + sizeof(uint32_t) * m_lowDetailGroupsCount;

  shaderData = m_shaderData;
}

size_t StreamingResident::cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex)
{
  TaskInfo& task = m_taskInfos[taskIndex];

  uint32_t taskOffset = m_maxGroups * taskIndex;

  task.region     = {};
  task.shaderData = m_shaderData;
  // we adjust the shaderio so that we skip the list of persistent data
  task.shaderData.activeGroupsCount   = m_activeGroupsCount - m_lowDetailGroupsCount;
  task.shaderData.activeClustersCount = m_activeClustersCount - m_lowDetailClustersCount;

  uint32_t deltaCount = m_groupIndicesUpdateRange.count();
  if(!deltaCount)
  {
    return 0;
  }

  // upload range of indices that was modified since last update
  // tightly packed in host

  memcpy(m_residentActiveHostBuffer.data() + taskOffset, m_activeGroupIndices.data() + m_groupIndicesUpdateRange.lo,
         sizeof(uint32_t) * deltaCount);

  // first copy from host to device update space now
  VkBufferCopy region;
  region.size      = sizeof(uint32_t) * deltaCount;
  region.srcOffset = sizeof(uint32_t) * taskOffset;
  region.dstOffset = m_residentActiveUpdateOffset + sizeof(uint32_t) * (taskOffset);
  vkCmdCopyBuffer(cmd, m_residentActiveHostBuffer.buffer, m_residentBuffer.buffer, 1, &region);

  // later we copy from update region to final persistent region
  task.region.size      = region.size;
  task.region.srcOffset = region.dstOffset;
  // though we apply the range offset for final region
  task.region.dstOffset = m_residentActiveOffset + sizeof(uint32_t) * m_groupIndicesUpdateRange.lo;

  // reset for next update
  m_groupIndicesUpdateRange = {};

  return region.size;
}

void StreamingResident::applyTask(shaderio::StreamingResident& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
  shaderData            = m_taskInfos[taskIndex].shaderData;
  shaderData.taskIndex  = taskIndex;
  shaderData.frameIndex = frameIndex;
}

void StreamingResident::cmdRunTask(VkCommandBuffer cmd, uint32_t taskIndex)
{
  TaskInfo& task = m_taskInfos[taskIndex];
  if(task.region.size)
  {
    vkCmdCopyBuffer(cmd, m_residentBuffer.buffer, m_residentBuffer.buffer, 1, &task.region);
  }
}

uint32_t StreamingResident::getLoadActiveGroupsOffset() const
{
  return m_activeGroupsCount - m_lowDetailGroupsCount;
}
uint32_t StreamingResident::getLoadActiveClustersOffset() const
{
  return m_activeClustersCount - m_lowDetailClustersCount;
}

bool StreamingResident::canAllocateGroup(uint32_t numClusters) const
{
  return m_groupAllocator.isAvailable(1, 1) && m_clusterAllocator.isAvailable(numClusters, 1);
}

const StreamingResident::Group* StreamingResident::findGroup(GeometryGroup geometryGroup) const
{
  auto it = m_mapGeometryGroup2Residency.find(geometryGroup.key);
  if(it == m_mapGeometryGroup2Residency.end())
  {
    return nullptr;
  }
  else
  {
    return &m_groups[it->second];
  }
}

StreamingResident::Group* StreamingResident::addGroup(GeometryGroup geometryGroup, uint32_t clusterCount)
{
  bool     valid = false;
  uint32_t groupResidentID;
  uint32_t clusterResidentID;
  uint32_t outAligned;
  uint32_t outSize;
  valid = m_groupAllocator.subAllocate(1, 1, groupResidentID, outAligned, outSize);
  assert(valid);
  valid = m_clusterAllocator.subAllocate(clusterCount, 1, clusterResidentID, outAligned, outSize);
  assert(valid);

  StreamingResident::Group& group = m_groups[groupResidentID];

  assert(m_mapGeometryGroup2Residency.find(geometryGroup.key) == m_mapGeometryGroup2Residency.end());
  m_mapGeometryGroup2Residency.insert({geometryGroup.key, groupResidentID});

  group.activeIndex       = m_activeGroupsCount++;
  group.geometryGroup     = geometryGroup;
  group.groupResidentID   = groupResidentID;
  group.clusterResidentID = clusterResidentID;
  group.clusterCount      = clusterCount;
  group.deviceAddress     = STREAMING_INVALID_ADDRESS_START;

  m_activeGroupIndices[group.activeIndex] = groupResidentID;

  // we don't need to do this, as Update task wile take care of modifying
  // the active groups buffer for all newly added
  // m_groupIndicesUpdateRange.update(group.activeIndex);

  m_activeClustersCount += clusterCount;

  return &m_groups[groupResidentID];
}

void StreamingResident::removeGroup(uint32_t groupResidentID)
{
  StreamingResident::Group& group = m_groups[groupResidentID];
  assert(m_mapGeometryGroup2Residency.find(group.geometryGroup.key) != m_mapGeometryGroup2Residency.end());
  m_mapGeometryGroup2Residency.erase(group.geometryGroup.key);

  {
    // remove group from compact indices list
    uint32_t activeIndex = group.activeIndex;

    // classic swapping our position in the active list with last element
    if(activeIndex + 1 != m_activeGroupsCount)
    {
      uint32_t lastResidentID              = m_activeGroupIndices[m_activeGroupsCount - 1];
      m_groups[lastResidentID].activeIndex = activeIndex;
      m_activeGroupIndices[activeIndex]    = lastResidentID;

      // we track those changes so that we later minimize the upload of
      // changed indices
      m_groupIndicesUpdateRange.update(activeIndex);
    }
    m_activeGroupsCount--;
  }

  m_activeClustersCount -= group.clusterCount;

  m_groupAllocator.subFree(groupResidentID, 1);
  m_clusterAllocator.subFree(group.clusterResidentID, group.clusterCount);

  group = {};
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingAllocator

void StreamingAllocator::init(Resources&                    res,
                              size_t                        totalMegaBytes,
                              uint32_t                      maxAllocationByteSize,
                              uint32_t                      granularityByteSize,
                              uint32_t                      sectorSizeShift,
                              shaderio::StreamingAllocator& shaderData)
{
  granularityByteSize = std::max(1u, granularityByteSize);

  // at least 2 warps
  assert(sectorSizeShift > 5 && granularityByteSize <= 0xFFFF);

  uint32_t granularityByteShift = 0;
  while((1u << granularityByteShift) < granularityByteSize && granularityByteShift <= 16)
  {
    granularityByteShift++;
  }
  // want power of two
  assert(granularityByteShift <= 16 && granularityByteSize == (1u << granularityByteShift));

  size_t sectorSize32s = size_t(1) << sectorSizeShift;
  size_t memoryBits    = size_t(totalMegaBytes) * 1024 * 1024 / granularityByteSize;
  size_t memory32s     = memoryBits / 32;
  size_t sectorCount   = memory32s / sectorSize32s;

  m_shaderData                      = {};
  m_shaderData.freeGapsCounter      = 0;
  m_shaderData.granularityByteShift = granularityByteShift;
  // align up to be multiple of 32
  m_shaderData.maxAllocationSize = (((maxAllocationByteSize + granularityByteSize - 1) / granularityByteSize) + 31) & (~31);
  m_shaderData.sectorSizeShift          = sectorSizeShift;
  m_shaderData.sectorMaxAllocationSized = uint32_t(sectorSize32s * 32 / m_shaderData.maxAllocationSize);
  m_shaderData.sectorCount              = uint32_t(sectorCount);

  // can only manage memory in multiple of sectorSize
  // so there might be some initial waste
  m_shaderData.baseWastedSize = uint32_t(memory32s - (sectorCount * sectorSize32s));

  // reset to multiples of sectors
  memory32s = sectorCount * sectorSize32s;

  BufferRanges ranges            = {};
  m_shaderData.freeGapsPos       = ranges.append(sizeof(uint32_t) * memory32s, 4);
  m_shaderData.freeGapsSize      = ranges.append(sizeof(uint16_t) * memory32s, 4);
  m_shaderData.freeGapsPosBinned = ranges.append(sizeof(uint32_t) * memory32s, 4);
  m_shaderData.freeSizeRanges    = ranges.append(sizeof(shaderio::AllocatorRange) * m_shaderData.maxAllocationSize, 8);
  m_shaderData.usedSectorBits    = ranges.append(sizeof(uint32_t) * ((sectorCount + 31) / 32), 4);
  m_shaderData.usedBits          = ranges.append(sizeof(uint32_t) * memory32s, 4);
  m_shaderData.stats             = ranges.append(sizeof(shaderio::AllocatorStats), 8);

  m_managementBuffer = res.createBuffer(ranges.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  m_shaderData.freeGapsPos += m_managementBuffer.address;
  m_shaderData.freeGapsSize += m_managementBuffer.address;
  m_shaderData.freeGapsPosBinned += m_managementBuffer.address;
  m_shaderData.freeSizeRanges += m_managementBuffer.address;
  m_shaderData.usedSectorBits += m_managementBuffer.address;
  m_shaderData.usedBits += m_managementBuffer.address;
  m_shaderData.stats += m_managementBuffer.address;

  m_shaderData.dispatchFreeGapsInsert.gridX = 1;
  m_shaderData.dispatchFreeGapsInsert.gridY = 1;
  m_shaderData.dispatchFreeGapsInsert.gridZ = 1;

  shaderData = m_shaderData;
}

void StreamingAllocator::deinit(Resources& res)
{
  res.destroy(m_managementBuffer);
}

size_t StreamingAllocator::getOperationsSize() const
{
  return m_managementBuffer.info.range;
}

uint32_t StreamingAllocator::getMaxSized() const
{
  return m_shaderData.sectorMaxAllocationSized * m_shaderData.sectorCount;
}

void StreamingAllocator::cmdReset(VkCommandBuffer cmd)
{
  vkCmdFillBuffer(cmd, m_managementBuffer.buffer, 0, m_managementBuffer.info.range, 0);
}

void StreamingAllocator::cmdBeginFrame(VkCommandBuffer cmd)
{
  // clears state to zero
  vkCmdFillBuffer(cmd, m_managementBuffer.buffer, m_shaderData.freeSizeRanges - m_managementBuffer.address,
                  sizeof(shaderio::AllocatorRange) * m_shaderData.maxAllocationSize, 0);
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingUpdates

void StreamingUpdates::init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment)
{
  m_clusterCountAlignment = clusterCountAlignment;
  m_scheduleIndex         = 0;
  m_pendingNew            = {};
  memset(m_scheduledNew, 0, sizeof(m_scheduledNew));
  memset(m_scheduledNewFrame, 0, sizeof(m_scheduledNewFrame));

  // some values are aligned up for easier gpu kernel access

  uint32_t loadRequests   = nvh::align_up(config.maxPerFrameLoadRequests, groupCountAlignment);
  uint32_t unloadRequests = nvh::align_up(config.maxPerFrameUnloadRequests, groupCountAlignment);

  m_shaderData                        = {};
  m_shaderData.patchGroupsCount       = loadRequests + unloadRequests;
  m_shaderData.patchUnloadGroupsCount = unloadRequests;

  m_unloadHandles = {};
  m_unloadHandles.resize(unloadRequests * STREAMING_MAX_ACTIVE_TASKS);

  std::vector<uint32_t> sharingQueueFamilies;
  if(config.useAsyncTransfer)
  {
    sharingQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);
    sharingQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);
  }

  res.createBufferTyped(m_patchesBuffer, m_shaderData.patchGroupsCount * STREAMING_MAX_ACTIVE_TASKS,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &sharingQueueFamilies);


  res.createBufferTyped(m_patchesHostBuffer, m_shaderData.patchGroupsCount * STREAMING_MAX_ACTIVE_TASKS, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  m_shaderData.patches = m_patchesBuffer.address;

  for(uint32_t c = 0; c < STREAMING_MAX_ACTIVE_TASKS; c++)
  {
    StreamingUpdates::TaskInfo& task = m_taskInfos[c];
    task.unloadPatches               = m_patchesHostBuffer.data() + m_shaderData.patchGroupsCount * c;
    task.loadPatches                 = task.unloadPatches + m_shaderData.patchUnloadGroupsCount;
    task.unloadHandles               = m_unloadHandles.data() + unloadRequests * c;
  }
}

size_t StreamingUpdates::getOperationsSize() const
{
  return m_patchesBuffer.info.range;
}

void StreamingUpdates::initClas(Resources& res, const StreamingConfig& config, const SceneConfig& sceneConfig)
{
  // some values are aligned up for easier gpu kernel access

  uint32_t maxLoadClusters = nvh::align_up(config.maxPerFrameLoadRequests * sceneConfig.clusterGroupSize, m_clusterCountAlignment);
  uint32_t maxClusters = nvh::align_up(config.maxClusters, m_clusterCountAlignment);

  BufferRanges ranges = {};

  m_shaderData.newClasBuilds      = ranges.append(sizeof(shaderio::ClasBuildInfo) * maxLoadClusters, 16);
  m_shaderData.newClasAddresses   = ranges.append(sizeof(uint64_t) * maxLoadClusters, 8);
  m_shaderData.newClasSizes       = ranges.append(sizeof(uint32_t) * maxLoadClusters, 4);
  m_shaderData.newClasResidentIDs = ranges.append(sizeof(uint32_t) * maxLoadClusters, 4);

  uint32_t maxMovedClusters = config.usePersistentClasAllocator ? maxLoadClusters : maxClusters;

  m_shaderData.moveClasDstAddresses = ranges.append(sizeof(uint64_t) * maxMovedClusters, 8);
  m_shaderData.moveClasSrcAddresses = ranges.append(sizeof(uint64_t) * maxMovedClusters, 8);

  m_clasBuffer = res.createBuffer(ranges.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  m_shaderData.newClasBuilds += m_clasBuffer.address;
  m_shaderData.newClasAddresses += m_clasBuffer.address;
  m_shaderData.newClasSizes += m_clasBuffer.address;
  m_shaderData.newClasResidentIDs += m_clasBuffer.address;

  m_shaderData.moveClasDstAddresses += m_clasBuffer.address;
  m_shaderData.moveClasSrcAddresses += m_clasBuffer.address;
}

size_t StreamingUpdates::getClasOperationsSize() const
{
  return m_clasBuffer.info.range;
}

void StreamingUpdates::deinitClas(Resources& res)
{
  res.destroy(m_clasBuffer);

  m_shaderData.newClasBuilds      = 0;
  m_shaderData.newClasAddresses   = 0;
  m_shaderData.newClasSizes       = 0;
  m_shaderData.newClasResidentIDs = 0;

  m_shaderData.moveClasDstAddresses = 0;
  m_shaderData.moveClasSrcAddresses = 0;
}

void StreamingUpdates::deinit(Resources& res)
{
  deinitClas(res);
  res.destroy(m_patchesBuffer);
  res.destroy(m_patchesHostBuffer);
}

void StreamingUpdates::reset()
{
  m_pendingNew = {};
  memset(m_scheduledNew, 0, sizeof(m_scheduledNew));
  memset(m_scheduledNewFrame, 0, sizeof(m_scheduledNewFrame));
  m_scheduleIndex = 0;
}

lodclusters::StreamingUpdates::TaskInfo& StreamingUpdates::getNewTask(uint32_t taskIndex)
{
  TaskInfo& task                = m_taskInfos[taskIndex];
  task.loadCount                = 0;
  task.unloadCount              = 0;
  task.newClusterCount          = 0;
  task.loadActiveGroupsOffset   = ~0;
  task.loadActiveClustersOffset = ~0;

  return task;
}

size_t StreamingUpdates::cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex)
{
  const TaskInfo& task = m_taskInfos[taskIndex];

  assert(task.loadActiveGroupsOffset != ~0);
  assert(task.loadActiveClustersOffset != ~0);

  size_t transferSize = 0;

  // copy from host buffer to device
  VkBufferCopy regions[2];
  uint32_t     regionCount = 0;

  regions[0].srcOffset = sizeof(shaderio::StreamingPatch) * m_shaderData.patchGroupsCount * taskIndex;
  regions[0].dstOffset = sizeof(shaderio::StreamingPatch) * m_shaderData.patchGroupsCount * taskIndex;

  if(task.unloadCount)
  {
    regions[0].size = sizeof(shaderio::StreamingPatch) * (task.unloadCount);
    regions[1]      = regions[0];
    regions[1].dstOffset += regions[0].size;

    transferSize += regions[0].size;

    regionCount++;
  }

  if(task.loadCount)
  {
    regions[regionCount].size = sizeof(shaderio::StreamingPatch) * (task.loadCount);
    regions[regionCount].srcOffset += sizeof(shaderio::StreamingPatch) * m_shaderData.patchUnloadGroupsCount;

    transferSize += regions[regionCount].size;

    regionCount++;
  }

  if(regionCount)
  {
    vkCmdCopyBuffer(cmd, m_patchesHostBuffer.buffer, m_patchesBuffer.buffer, regionCount, regions);
  }

  // we know this task will get scheduled eventually
  m_pendingNew.clusters += task.newClusterCount;
  m_pendingNew.groups += task.loadCount;

  return transferSize;
}

void StreamingUpdates::applyTask(shaderio::StreamingUpdate& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
  const TaskInfo& task = m_taskInfos[taskIndex];
  // keep basics
  shaderData = m_shaderData;
  // override counts
  shaderData.patchGroupsCount       = task.loadCount + task.unloadCount;
  shaderData.patchUnloadGroupsCount = task.unloadCount;
  shaderData.newClasCount           = task.newClusterCount;

  // adjust pointer offset
  shaderData.patches += sizeof(shaderio::StreamingPatch) * m_shaderData.patchGroupsCount * taskIndex;
  shaderData.taskIndex                = taskIndex;
  shaderData.frameIndex               = frameIndex;
  shaderData.loadActiveGroupsOffset   = task.loadActiveGroupsOffset;
  shaderData.loadActiveClustersOffset = task.loadActiveClustersOffset;

  // we also want to keep track of the total amount of "future" cluster builds.
  // This is relevant to ray tracing, as the GPU's allocator need to provide enough
  // space for this number of "worst case" cluster or group sizes.
  assert(m_pendingNew.clusters >= task.newClusterCount);
  assert(m_pendingNew.groups >= task.loadCount);

  m_pendingNew.clusters -= task.newClusterCount;
  m_pendingNew.groups -= task.loadCount;
  m_scheduledNewFrame[m_scheduleIndex % STREAMING_MAX_ACTIVE_TASKS]     = frameIndex;
  m_scheduledNew[m_scheduleIndex % STREAMING_MAX_ACTIVE_TASKS].clusters = task.newClusterCount;
  m_scheduledNew[m_scheduleIndex % STREAMING_MAX_ACTIVE_TASKS].groups   = task.loadCount;
  m_scheduleIndex++;
}

//////////////////////////////////////////////////////////////////////////
//
// StreamingGeometry

void StreamingStorage::init(Resources& res, const StreamingConfig& config)
{
  m_maxSceneBytes    = config.maxGeometryMegaBytes * 1024 * 1024;
  m_maxTransferBytes = config.maxTransferMegaBytes * 1024 * 1024;
  m_blockBytes       = std::min(size_t(128) * 1024 * 1024, m_maxSceneBytes);

  std::vector<uint32_t> sharingQueueFamilies;
  if(config.useAsyncTransfer)
  {
    sharingQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);
    sharingQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);

    m_taskCommandPool.init(res.m_device, res.m_queueStates.transfer.m_familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                           STREAMING_MAX_ACTIVE_TASKS);
  }

  m_transferHostBuffer = res.createBuffer(m_maxTransferBytes * STREAMING_MAX_ACTIVE_TASKS, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dataAllocator.init(res.m_allocator.getMemoryAllocator(), m_blockBytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, sharingQueueFamilies);

  m_copyRegions = {};
  m_copyInfos   = {};
  m_copyRegions.reserve(config.maxGroups);
  m_copyInfos.reserve(config.maxGroups);
}

void StreamingStorage::deinit(Resources& res)
{
  res.destroy(m_transferHostBuffer);
  m_dataAllocator.deinit();
  m_taskCommandPool.deinit();

  m_copyInfos   = {};
  m_copyRegions = {};
}

size_t StreamingStorage::getOperationsSize() const
{
  // the geometry storage is not tracked as fixed operations
  return 0;
}

size_t StreamingStorage::getMaxDataSize() const
{
  return (m_maxSceneBytes / m_blockBytes) * m_blockBytes;
}

lodclusters::StreamingStorage::TaskInfo& StreamingStorage::getNewTask(uint32_t taskIndex)
{
  TaskInfo& task  = m_taskOperations[taskIndex];
  task.baseOffset = m_maxTransferBytes * taskIndex;
  task.usedMemory = 0;

  m_copyInfos.clear();
  m_copyRegions.clear();

  return task;
}

bool StreamingStorage::canTransfer(const TaskInfo& task, size_t size) const
{
  return task.usedMemory + size <= m_maxTransferBytes;
}

void* StreamingStorage::appendTransfer(TaskInfo& task, nvvk::BufferSubAllocator::Handle dstHandle)
{
  nvvk::BufferSubAllocator::Binding dstBinding = m_dataAllocator.getSubBinding(dstHandle);

  assert(task.usedMemory + dstBinding.size <= m_maxTransferBytes);

  size_t transferOffset  = task.baseOffset;
  void*  transferPointer = reinterpret_cast<uint8_t*>(m_transferHostBuffer.mapping) + task.baseOffset;

  task.usedMemory += dstBinding.size;
  task.baseOffset += dstBinding.size;

  if(!m_copyInfos.empty() && m_copyInfos.back().targetBuffer == dstBinding.buffer)
  {
    VkBufferCopy& lastRegion = m_copyRegions.back();

    // check if we can grow the last region
    if(lastRegion.dstOffset + lastRegion.size == dstBinding.offset)
    {
      lastRegion.size += dstBinding.size;
      return transferPointer;
    }

    // otherwise append new region below
  }
  else
  {
    // new target buffer
    CopyInfo task;
    task.targetBuffer = dstBinding.buffer;
    task.regionOffset = m_copyRegions.size();
    task.regionCount  = 0;
    m_copyInfos.push_back(task);
  }

  {
    // append new region
    VkBufferCopy region;
    region.srcOffset = transferOffset;
    region.dstOffset = dstBinding.offset;
    region.size      = dstBinding.size;

    m_copyInfos.back().regionCount++;
    m_copyRegions.push_back(region);
  }

  return transferPointer;
}

uint32_t StreamingStorage::cmdUploadTask(VkCommandBuffer cmd)
{
  for(auto it : m_copyInfos)
  {
    vkCmdCopyBuffer(cmd, m_transferHostBuffer.buffer, it.targetBuffer, uint32_t(it.regionCount), &m_copyRegions[it.regionOffset]);
  }

  return uint32_t(m_copyRegions.size());
}

bool StreamingStorage::canAllocate(size_t sz) const
{
  VkDeviceSize allocatedSize;
  VkDeviceSize usedSize;
  m_dataAllocator.getUtilization(allocatedSize, usedSize);

  return m_dataAllocator.fitsInAllocated(sz) || (allocatedSize + m_blockBytes) <= m_maxSceneBytes;
}

void StreamingStorage::reset()
{
  m_dataAllocator.free(false);
}

nvvk::BufferSubAllocator::Handle StreamingStorage::allocate(GeometryGroup geometryGroup, size_t sz, uint64_t& deviceAddress)
{
  nvvk::BufferSubAllocator::Handle handle = m_dataAllocator.subAllocate(sz);

  deviceAddress = m_dataAllocator.getSubBinding(handle).address;
  assert(handle.isValid());
  return handle;
}

void StreamingStorage::free(nvvk::BufferSubAllocator::Handle handle)
{
  assert(handle.isValid());
  m_dataAllocator.subFree(handle);
}

void StreamingStorage::getStats(StreamingStats& stats) const
{
  m_dataAllocator.getUtilization(stats.reservedDataBytes, stats.usedDataBytes);
}

}  // namespace lodclusters
