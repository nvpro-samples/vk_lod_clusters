/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>

#include <volk.h>

#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "async_uploader.hpp"

namespace {

bool isPowerOfTwo(VkDeviceSize value)
{
  return value > 0 && (value & (value - 1)) == 0;
}

uint64_t setRangeBits(uint32_t startBit, uint32_t endBit)
{
  assert(startBit <= endBit && endBit < 64);
  uint64_t mask = 0;
  for(uint32_t bit = startBit; bit <= endBit; bit++)
  {
    mask |= (1ull << bit);
  }
  return mask;
}

}  // namespace


AsyncUploader::~AsyncUploader()
{
  assert(m_info.allocator == nullptr && "Missing deinit()");
}

VkResult AsyncUploader::init(const InitInfo& info)
{
  assert(m_info.allocator == nullptr && "Missing deinit()");
  m_info = info;

  VkResult result;
  VkDevice device = info.allocator->getDevice();

  result = nvvk::createTimelineSemaphore(device, 0, m_transferTimelineSemaphore);
  if(result != VK_SUCCESS)
  {
    return result;
  }

  result = m_transferCmdPool.init(device, info.transferQueue.familyIndex, nvvk::ManagedCommandPools::Mode::SEMAPHORE_STATE,
                                  VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, info.commandPoolSize);
  if(result != VK_SUCCESS)
  {
    vkDestroySemaphore(device, m_transferTimelineSemaphore, nullptr);
    return result;
  }

  nvvk::BufferSubAllocator::InitInfo subInitInfo;
  subInitInfo.allocationFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  subInitInfo.memoryUsage = info.forceCoherentMapping ? VMA_MEMORY_USAGE_CPU_ONLY : VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  subInitInfo.usageFlags  = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;
  subInitInfo.resourceAllocator     = info.allocator;
  subInitInfo.debugName             = info.debugName;
  subInitInfo.blockSize             = info.blockSize;
  subInitInfo.maxAllocatedSize      = info.blockSize * MAX_FLUSH_BLOCKS;
  subInitInfo.perBlockAllocations   = info.maxOperations;
  subInitInfo.keepBlockCount        = info.keepBlockCount;
  subInitInfo.threadSafeBlockAccess = true;

  m_flushRangeSize = info.blockSize / RANGES_PER_BLOCK;
  assert(info.blockSize % RANGES_PER_BLOCK == 0);

  result = m_bufferSubAllocator.init(subInitInfo);
  if(result != VK_SUCCESS)
  {
    vkDestroySemaphore(device, m_transferTimelineSemaphore, nullptr);
    m_transferCmdPool.deinit();
    m_transferTimelineSemaphore = nullptr;
    return result;
  }

  m_batch.enableLayoutBarriers = true;
  m_batch.enableOwnerBarriers  = m_info.transferQueue.queue != m_info.targetQueue.queue;
  m_batch.dstQueueFamilyIndex  = info.targetQueue.familyIndex;
  m_batch.srcQueueFamilyIndex  = info.transferQueue.familyIndex;

  m_transferSubmitState = nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue);

  m_mappingAllocations.resize(info.maxOperations);
  m_mappingPool.init(info.maxOperations);

  return VK_SUCCESS;
}

void AsyncUploader::deinit()
{
  if(m_info.allocator == nullptr)
    return;

  VkDevice device = m_info.allocator->getDevice();
  NVVK_CHECK(vkQueueWaitIdle(m_info.transferQueue.queue));

  releaseStagingAllocations(true);

  for(size_t i = 0; i < m_mappingAllocations.size(); i++)
  {
    if(m_mappingAllocations[i])
    {
      m_bufferSubAllocator.subFree(m_mappingAllocations[i]);
      m_mappingPool.destroyID(uint32_t(i));
    }
  }

  assert(m_stagingAllocations.empty() && m_stagingAllocationsSize == 0);
  m_batch.reset();
  vkDestroySemaphore(device, m_transferTimelineSemaphore, nullptr);
  m_transferTimelineSemaphore = nullptr;
  m_transferCmdPool.deinit();
  m_bufferSubAllocator.deinit();
  m_info               = {};
  m_mappingAllocations = {};
  m_mappingPool.deinit();
  m_transferSubmitState   = {};
  m_transferTimelineValue = 1;
  m_submitCount           = 0;
  m_waitCmdCount          = 0;
  m_targetBarriers.clear();
  m_batchFlushRangeMasks = {};
  m_allocationCount      = 0;
}

VkResult AsyncUploader::submitBatch()
{
  VkCommandBuffer cmd;
  NVVK_FAIL_RETURN(m_transferCmdPool.acquireCommandBuffer(m_transferSubmitState, cmd));
  NVVK_DBG_CUSTOM_NAME(cmd, "AsyncUploader:cmd:" + std::to_string(m_transferTimelineValue));

  releaseStagingAllocations(false);

  // Snapshot the ownership-acquisition barriers before cmdCopyAppended() consumes the batch. The
  // record is only published to m_targetBarriers after a successful submit (below), so a failed
  // vkQueueSubmit2() leaves no record whose semaphore would never be signaled.
  TargetBarriers record{
      .ownerAcquisitionBarriers = std::move(m_batch.acquire),
      .semaphoreState           = m_transferSubmitState,
  };

  VkCommandBufferBeginInfo cmdBegin{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(cmd, &cmdBegin);

  void* payload = nullptr;
  if(m_info.onSubmitCallback)
  {
    payload = m_info.onSubmitCallback(cmd, m_submitCount, m_transferSubmitState, true, nullptr);
  }
  if(!m_info.forceCoherentMapping)
  {
    flushBatchBlocks();
  }
  m_batch.cmdCopyAppended(cmd);
  if(m_info.onSubmitCallback)
  {
    m_info.onSubmitCallback(cmd, m_submitCount, m_transferSubmitState, false, payload);
  }
  vkEndCommandBuffer(cmd);
  m_waitCmdCount = m_transferCmdPool.getWaitCount();
  m_submitCount++;

  VkCommandBufferSubmitInfo cmdSubmitInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmdSubmitInfo.commandBuffer             = cmd;

  VkSemaphoreSubmitInfo semSubmitInfo = nvvk::makeSemaphoreSubmitInfo(m_transferSubmitState, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  // prepare actual submit
  VkSubmitInfo2 submitInfo2            = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submitInfo2.commandBufferInfoCount   = 1;
  submitInfo2.pCommandBufferInfos      = &cmdSubmitInfo;
  submitInfo2.signalSemaphoreInfoCount = 1;
  submitInfo2.pSignalSemaphoreInfos    = &semSubmitInfo;

  // submit to queue
  NVVK_FAIL_RETURN(vkQueueSubmit2(m_info.transferQueue.queue, 1, &submitInfo2, VK_NULL_HANDLE));

  // submit succeeded: now retain the ownership-barrier record for cmdDrainOwnershipBarriers()
  {
    std::lock_guard lock(m_targetLock);
    m_targetBarriers.push_back(std::move(record));
  }

  m_transferTimelineValue++;
  m_transferSubmitState = nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue);

  return VK_SUCCESS;
}

void AsyncUploader::releaseStagingAllocations(bool forceAll)
{
  VkDevice device = m_info.allocator->getDevice();

  nvvk::SemaphoreStateSignalCache semaphoreCache;

  size_t writeIndex = 0;

  for(size_t readIndex = 0; readIndex < m_stagingAllocations.size(); readIndex++)
  {
    StagingAllocation& allocation = m_stagingAllocations[readIndex];

    if(forceAll || !allocation.semaphoreState.isValid() || (semaphoreCache.testSignaled(device, allocation.semaphoreState)))
    {
      std::lock_guard lock(m_allocLock);

      nvvk::BufferRange range = m_bufferSubAllocator.subRange(allocation.subAllocation);
      m_stagingAllocationsSize -= range.range;

      m_bufferSubAllocator.subFree(allocation.subAllocation);
      m_allocationCount--;
    }
    else if(readIndex != writeIndex)
    {
      m_stagingAllocations[writeIndex++] = allocation;
    }
    else
    {
      writeIndex++;
    }
  }
  m_stagingAllocations.resize(writeIndex);
}

VkResult AsyncUploader::submitBatchChecked(size_t addedSize)
{
  if(m_batch.checkAppendedSize(m_info.flushSize, addedSize))
  {
    return submitBatch();
  }
  return VK_SUCCESS;
}

bool AsyncUploader::consumeMappingHandle(MappingHandle handle, nvvk::BufferSubAllocation& subAllocation)
{
  MappingHandleDetail detail{};
  detail.handle        = handle;
  const uint32_t index = detail.mappingIndex;
  assert(detail.valid != 0);

  // copy subAllocation before we destroy the ID, as otherwise
  // someone else can overwrite the slot
  subAllocation               = m_mappingAllocations[index];
  m_mappingAllocations[index] = {};

  bool result;
  {
    std::lock_guard lock(m_mappingLock);
    result = m_mappingPool.destroyID(index);
  }
  if(!result)
    return false;

  return true;
}

void AsyncUploader::markRangeForFlush(uint32_t blockIndex, VkDeviceSize offset, VkDeviceSize size)
{
  if(m_info.forceCoherentMapping || size == 0)
  {
    return;
  }

  assert(blockIndex < MAX_FLUSH_BLOCKS && "AsyncUploader supports at most 64 blocks");


  if(offset + size > m_flushRangeSize * RANGES_PER_BLOCK)
  {
    m_batchFlushRangeMasks[blockIndex] = ALL_RANGE_BITS;
    return;
  }

  const VkDeviceSize rangeSize = m_flushRangeSize;
  const uint32_t     startBit  = static_cast<uint32_t>(offset / rangeSize);
  const uint32_t     endBit    = static_cast<uint32_t>((offset + size - 1) / rangeSize);

  assert(endBit < RANGES_PER_BLOCK);
  m_batchFlushRangeMasks[blockIndex] |= setRangeBits(startBit, endBit);
}

void AsyncUploader::markForFlush(const nvvk::BufferSubAllocation& subAllocation)
{
  if(m_info.forceCoherentMapping || !subAllocation)
  {
    return;
  }

  markRangeForFlush(subAllocation.getBlockIndex(), subAllocation.getOffset(m_bufferSubAllocator.getOffsetUnitSize()),
                    subAllocation.getSize());
}

void AsyncUploader::flushBatchBlocks()
{
  for(uint32_t blockIndex = 0; blockIndex < MAX_FLUSH_BLOCKS; blockIndex++)
  {
    const uint64_t rangeMask = m_batchFlushRangeMasks[blockIndex];
    if(rangeMask == 0)
    {
      continue;
    }

    const nvvk::Buffer& blockBuffer = m_bufferSubAllocator.getBlockBuffer(static_cast<uint16_t>(blockIndex));
    const VkDeviceSize  blockSize   = blockBuffer.bufferSize;

    if(rangeMask == ALL_RANGE_BITS || !isPowerOfTwo(blockSize))
    {
      m_info.allocator->autoFlushBuffer(blockBuffer);
      continue;
    }

    const VkDeviceSize rangeSize = blockSize / RANGES_PER_BLOCK;

    for(uint32_t bit = 0; bit < RANGES_PER_BLOCK; bit++)
    {
      if((rangeMask & (1ull << bit)) == 0)
      {
        continue;
      }

      const uint32_t startBit = bit;
      while(bit + 1 < RANGES_PER_BLOCK && (rangeMask & (1ull << (bit + 1))))
      {
        bit++;
      }

      const VkDeviceSize flushOffset = VkDeviceSize(startBit) * rangeSize;
      const VkDeviceSize flushSize   = VkDeviceSize(bit - startBit + 1) * rangeSize;
      m_info.allocator->autoFlushBuffer(blockBuffer, flushOffset, flushSize);
    }
  }

  m_batchFlushRangeMasks = {};
}

VkResult AsyncUploader::acquireStagingSpace(nvvk::BufferSubAllocation& subAllocation,
                                            nvvk::BufferRange&         stagingSpace,

                                            size_t      dataSize,
                                            const void* data)
{
  VkResult result = m_bufferSubAllocator.subAllocate(subAllocation, dataSize);
  if(subAllocation.getBlockIndex() >= MAX_FLUSH_BLOCKS)
  {
    assert(0 && "AsyncUploader supports at most 64 blocks");
    return VK_ERROR_OUT_OF_POOL_MEMORY;
  }
  if(result == VK_SUCCESS)
  {
    stagingSpace = m_bufferSubAllocator.subRange(subAllocation);
    m_stagingAllocationsSize += stagingSpace.range;
  }
  m_allocationCount++;

  return result;
}

void AsyncUploader::addStagingAllocation(const nvvk::BufferSubAllocation& subAllocation)
{
  StagingAllocation stagingAllocation = {};
  stagingAllocation.semaphoreState    = m_transferSubmitState;
  stagingAllocation.subAllocation     = subAllocation;
  m_stagingAllocations.push_back(stagingAllocation);
}

VkResult AsyncUploader::acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle)
{
  mappingHandle = 0;

  if(dataSize == 0)
  {
    mappingSpace = {};
    return VK_SUCCESS;
  }
  uint32_t mappingIndex;
  {
    std::lock_guard lock(m_mappingLock);
    if(!m_mappingPool.createID(mappingIndex))
      return VK_ERROR_UNKNOWN;
  }

  VkResult result;
  {
    std::lock_guard lock(m_allocLock);
    result = acquireStagingSpace(m_mappingAllocations[mappingIndex], mappingSpace, dataSize, nullptr);
  }

  if(result == VK_SUCCESS)
  {
    MappingHandleDetail detail{};
    detail.valid        = 1;
    detail.mappingIndex = mappingIndex;
    detail.blockIndex   = m_mappingAllocations[mappingIndex].getBlockIndex();
    mappingHandle       = detail.handle;
    return VK_SUCCESS;
  }

  {
    std::lock_guard lock(m_mappingLock);
    m_mappingPool.destroyID(mappingIndex);
  }
  return result;
}

void AsyncUploader::releaseMapping(MappingHandle mappingHandle)
{
  if(mappingHandle == 0)
  {
    return;
  }

  MappingHandleDetail detail{};
  detail.handle               = mappingHandle;
  const uint32_t mappingIndex = detail.mappingIndex;
  assert(detail.valid != 0);

  {
    std::lock_guard lock(m_allocLock);

    m_stagingAllocationsSize -= m_mappingAllocations[mappingIndex].getSize();
    m_bufferSubAllocator.subFree(m_mappingAllocations[mappingIndex]);
    m_mappingAllocations[mappingIndex] = {};
  }
  {
    std::lock_guard lock(m_mappingLock);

    m_mappingPool.destroyID(mappingIndex);
  }
}

VkResult AsyncUploader::appendBufferRange(const nvvk::BufferRange& buffer, const void* data, nvvk::SemaphoreState* outSemaphoreState)
{
  if(buffer.range == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return VK_SUCCESS;
  }

  assert(buffer.buffer);
  assert(data);

  nvvk::BufferSubAllocation subAllocation;
  nvvk::BufferRange         stagingSpace;
  {
    std::lock_guard lock(m_allocLock);
    NVVK_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, buffer.range, nullptr));
  }

  memcpy(stagingSpace.mapping, data, buffer.range);

  {
    std::lock_guard lock(m_appendLock);
    NVVK_FAIL_RETURN(submitBatchChecked(buffer.range));

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, buffer.offset, buffer.range, true);
    addStagingAllocation(subAllocation);

    markForFlush(subAllocation);

    setOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

VkResult AsyncUploader::appendBufferRangeMappings(size_t                    rangeCount,
                                                  const MappedBufferRanges* ranges,
                                                  MappingHandle             mappingHandle,
                                                  bool                      releaseMapping_,
                                                  nvvk::SemaphoreState*     outSemaphoreState)
{
  nvvk::BufferSubAllocation subAllocation{};
  nvvk::BufferRange         fullMappingSpace{};
  bool                      consumedHandle = false;

  assert(mappingHandle);

  if(releaseMapping_)
  {
    if(!consumeMappingHandle(mappingHandle, subAllocation))
    {
      return VK_ERROR_UNKNOWN;
    }
    consumedHandle = true;
  }

  size_t totalSize = 0;
  for(size_t i = 0; i < rangeCount; i++)
  {
    const MappedBufferRanges& range = ranges[i];
    totalSize += range.bufferRange.range;
  }

  {
    std::lock_guard lock(m_appendLock);

    if(totalSize > 0)
    {
      NVVK_FAIL_RETURN(submitBatchChecked(totalSize));
    }

    MappingHandleDetail detail{};
    detail.handle = mappingHandle;

    const uint32_t mappingBlockIndex = consumedHandle ? subAllocation.getBlockIndex() : detail.blockIndex;

    for(size_t i = 0; i < rangeCount; i++)
    {
      const MappedBufferRanges& range = ranges[i];
      if(range.bufferRange.range == 0)
        continue;

      assert(range.bufferRange.buffer);
      assert(range.mappingRange.range == range.bufferRange.range);

      markRangeForFlush(mappingBlockIndex, range.mappingRange.offset, range.mappingRange.range);

      totalSize += range.bufferRange.range;
      m_batch.addBufferCopy(range.mappingRange.buffer, range.mappingRange.offset, range.bufferRange.buffer,
                            range.bufferRange.offset, range.bufferRange.range);
    }

    if(consumedHandle)
    {
      if(totalSize > 0)
      {
        addStagingAllocation(subAllocation);
      }
      else
      {
        std::lock_guard lock(m_allocLock);
        m_stagingAllocationsSize -= subAllocation.getSize();
        m_bufferSubAllocator.subFree(subAllocation);
        m_allocationCount--;
      }
    }

    setOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

VkResult AsyncUploader::appendImageSub(nvvk::Image&                    image,
                                       const VkOffset3D&               offset,
                                       const VkExtent3D&               extent,
                                       const VkImageSubresourceLayers& subresource,
                                       size_t                          dataSize,
                                       const void*                     data,
                                       VkImageLayout                   newLayout,
                                       nvvk::SemaphoreState*           outSemaphoreState)
{
  if(dataSize == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return VK_SUCCESS;
  }

  assert(image.image);
  assert(data);

  nvvk::BufferSubAllocation subAllocation;
  nvvk::BufferRange         stagingSpace;
  {
    std::lock_guard lock(m_allocLock);
    NVVK_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, dataSize, nullptr));
  }

  memcpy(stagingSpace.mapping, data, dataSize);

  const VkImageSubresourceRange subresourceRange{subresource.aspectMask, subresource.mipLevel, 1,
                                                 subresource.baseArrayLayer, subresource.layerCount};

  {
    std::lock_guard lock(m_appendLock);
    NVVK_FAIL_RETURN(submitBatchChecked(dataSize));

    m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image.image, image.descriptor.imageLayout, newLayout,
                         dataSize, subresource, offset, extent, &subresourceRange);
    addStagingAllocation(subAllocation);
    markForFlush(subAllocation);

    setOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

VkResult AsyncUploader::appendImageSubMappings(nvvk::Image&           image,
                                               size_t                 imageSubCount,
                                               const MappedImageSubs* imageSubs,
                                               MappingHandle          mappingHandle,
                                               bool                   releaseMapping_,
                                               VkImageLayout          newLayout,
                                               nvvk::SemaphoreState*  outSemaphoreState)
{
  assert(image.image);
  assert(mappingHandle);

  nvvk::BufferSubAllocation subAllocation{};
  bool                      consumedHandle = false;

  if(releaseMapping_)
  {
    if(!consumeMappingHandle(mappingHandle, subAllocation))
    {
      return VK_ERROR_UNKNOWN;
    }
    consumedHandle = true;
  }

  size_t totalSize = 0;
  for(size_t i = 0; i < imageSubCount; i++)
  {
    const MappedImageSubs& sub = imageSubs[i];
    totalSize += sub.mappingSpace.range;
  }

  {
    VkImageLayout imageLayout = image.descriptor.imageLayout;

    std::lock_guard lock(m_appendLock);

    if(totalSize > 0)
    {
      NVVK_FAIL_RETURN(submitBatchChecked(totalSize));
    }

    MappingHandleDetail detail{};
    detail.handle = mappingHandle;

    const uint32_t mappingBlockIndex = consumedHandle ? subAllocation.getBlockIndex() : detail.blockIndex;

    for(size_t i = 0; i < imageSubCount; i++)
    {
      const MappedImageSubs& sub = imageSubs[i];
      if(sub.mappingSpace.range == 0)
        continue;

      markRangeForFlush(mappingBlockIndex, sub.mappingSpace.offset, sub.mappingSpace.range);

      // reset imageLayout with each subImage
      image.descriptor.imageLayout = imageLayout;

      const VkImageSubresourceRange subresourceRange{sub.subresource.aspectMask, sub.subresource.mipLevel, 1,
                                                     sub.subresource.baseArrayLayer, sub.subresource.layerCount};

      m_batch.addImageCopy(sub.mappingSpace.buffer, sub.mappingSpace.offset, image.image, image.descriptor.imageLayout,
                           newLayout, sub.mappingSpace.range, sub.subresource, sub.offset, sub.extent, &subresourceRange);
    }

    if(consumedHandle)
    {
      if(totalSize > 0)
      {
        addStagingAllocation(subAllocation);
      }
      else
      {
        std::lock_guard lock(m_allocLock);
        m_stagingAllocationsSize -= subAllocation.getSize();
        m_bufferSubAllocator.subFree(subAllocation);
        m_allocationCount--;
      }
    }

    setOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

nvvk::BufferSubAllocator::Report AsyncUploader::getAllocationReport() const
{
  std::lock_guard lock(m_allocLock);

  return m_bufferSubAllocator.getReport();
}

bool AsyncUploader::hasOwnershipBarriers() const
{
  std::lock_guard lock(m_targetLock);

  return !m_targetBarriers.empty();
}

VkResult AsyncUploader::flushPending(nvvk::SemaphoreState* outSemaphoreState)
{
  std::lock_guard lock(m_appendLock);

  if(!m_batch.isAppendedEmpty())
  {
    setOutSemaphoreState(outSemaphoreState);
    return submitBatch();
  }
  else
  {
    if(outSemaphoreState)
    {
      *outSemaphoreState = nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue - 1);
    }
    releaseStagingAllocations(false);
    return VK_SUCCESS;
  }
}

VkResult AsyncUploader::waitForCompletion()
{
  if(m_transferTimelineValue == 1)
    return VK_SUCCESS;

  // current value is always for next submit, hence -1
  nvvk::SemaphoreState sem = nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue - 1);
  return sem.wait(m_info.allocator->getDevice(), ~0ull);
}

void AsyncUploader::releaseCompletedAllocations()
{
  std::lock_guard lock(m_appendLock);
  releaseStagingAllocations(false);
}

nvvk::SemaphoreState AsyncUploader::getSubmitState()
{
  std::lock_guard lock(m_appendLock);

  if(!m_batch.isAppendedEmpty())
  {
    return m_transferSubmitState;
  }
  else
  {
    return nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue - 1);
  }
}

void AsyncUploader::cmdDrainOwnershipBarriers(VkCommandBuffer cmd)
{
  std::lock_guard lock(m_targetLock);

  if(m_targetBarriers.empty())
    return;

  VkDevice device = m_info.allocator->getDevice();

  nvvk::SemaphoreStateSignalCache semaphoreCache;

  size_t writeIndex = 0;

  for(size_t readIndex = 0; readIndex < m_targetBarriers.size(); readIndex++)
  {
    TargetBarriers& barriers = m_targetBarriers[readIndex];

    if(!barriers.semaphoreState.isValid() || (semaphoreCache.testSignaled(device, barriers.semaphoreState)))
    {
      barriers.ownerAcquisitionBarriers.cmdPipelineBarrier(cmd, 0);
      barriers.ownerAcquisitionBarriers.clear();
    }
    else if(readIndex != writeIndex)
    {
      m_targetBarriers[writeIndex++] = std::move(barriers);
    }
    else
    {
      writeIndex++;
    }
  }
  m_targetBarriers.resize(writeIndex);
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_AsyncUploader()
{
  // AsyncUploader submits copy batches on a dedicated transfer queue.
  // Before using uploaded resources on the graphics queue, call cmdDrainOwnershipBarriers().

  nvvk::ResourceAllocator resourceAllocator;  // EX. initialize somehow
  nvvk::QueueInfo         transferQueue{};    // EX. dedicated transfer queue with exclusive submit access
  nvvk::QueueInfo         graphicsQueue{};    // EX. queue where uploaded resources are consumed
  nvvk::Buffer            deviceBuffer{};     // EX. create GPU buffer to upload into
  nvvk::Image             deviceImage{};      // EX. create GPU image to upload into

  AsyncUploader asyncUploader;
  asyncUploader.init({
      .allocator     = &resourceAllocator,
      .transferQueue = transferQueue,
      .targetQueue   = graphicsQueue,
      .debugName     = "asyncUploads",
  });

  uint8_t uploadData[4096]{};
  size_t  uploadSize = sizeof(uploadData);

  // worker thread(s) can enqueue uploads concurrently
  {
    asyncUploader.appendBuffer(deviceBuffer, 0, uploadSize, uploadData);

    // mapping upload: acquire staging, fill on CPU, append, then flush batch to transfer queue
    {
      nvvk::BufferRange                mappingSpace{};
      UploaderInterface::MappingHandle mappingHandle{};
      if(asyncUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle) == VK_SUCCESS)
      {
        memcpy(mappingSpace.mapping, uploadData, uploadSize);

        UploaderInterface::MappedBufferRanges mappedRange{};
        mappedRange.bufferRange.buffer = deviceBuffer.buffer;
        mappedRange.bufferRange.offset = 0;
        mappedRange.bufferRange.range  = uploadSize;
        mappedRange.mappingRange       = mappingSpace;

        nvvk::SemaphoreState uploadSem{};
        asyncUploader.appendBufferRangeMappings(1, &mappedRange, mappingHandle, true, &uploadSem);
      }
    }

    // cancel path: release mapping if acquired but not submitted
    {
      nvvk::BufferRange                mappingSpace{};
      UploaderInterface::MappingHandle mappingHandle{};
      if(asyncUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle) == VK_SUCCESS)
      {
        asyncUploader.releaseMapping(mappingHandle);
      }
    }

    // direct image upload
    {
      VkOffset3D               offset{};
      VkExtent3D               extent{64, 64, 1};
      VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      asyncUploader.appendImageSub(deviceImage, offset, extent, subresource, uploadSize, uploadData,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    nvvk::SemaphoreState flushSem{};
    asyncUploader.flushPending(&flushSem);

    (void)asyncUploader.getSubmitState();
    (void)asyncUploader.hasOwnershipBarriers();
    (void)asyncUploader.getAllocationReport();
  }

  while(true)
  {
    VkCommandBuffer cmd{};  // per-frame graphics command buffer

    // apply ownership barriers before using async-uploaded resources
    asyncUploader.cmdDrainOwnershipBarriers(cmd);

    asyncUploader.releaseCompletedAllocations();
  }

  // call asyncUploader.waitForCompletion() and asyncUploader.deinit() on shutdown
}
