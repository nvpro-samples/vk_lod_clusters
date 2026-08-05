/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/buffer_suballocator.hpp>
#include <nvvk/command_pools.hpp>
#include <nvutils/id_pool.hpp>
#include <functional>
#include <array>

#include "uploader_interface.hpp"

class AsyncUploader : public UploaderInterface
{
public:
  AsyncUploader() = default;
  ~AsyncUploader();
  AsyncUploader(const AsyncUploader&)            = delete;
  AsyncUploader& operator=(const AsyncUploader&) = delete;

  struct InitInfo
  {
    nvvk::ResourceAllocator* allocator{};
    // the uploader must have exclusive submit access to this queue,
    // while being used.
    nvvk::QueueInfo transferQueue{};

    // The target queue for where the resources are used at the end.
    // One must call `cmdDrainOwnershipBarriers` on this queue, prior
    // using resources that completed their transfer.
    //
    // If targetQueue.queue and transferQueue.queue match, the ownership
    // transfer is skipped.
    nvvk::QueueInfo targetQueue{};

    // how many command buffers can be in flight
    uint32_t commandPoolSize = 3;

    // after how much appended staging memory we trigger a flush
    uint32_t flushSize = 64 * 1024 * 1024;

    VkDeviceSize blockSize = 128 * 1024 * 1024;

    // keeping at least one block is recommended for persistent usage
    uint32_t keepBlockCount = 1;

    // number of copy operations per block
    // number of mappings in flight
    uint32_t    maxOperations = 0xFFFF;
    std::string debugName{};

    // if true, staging memory is guaranteed to have VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    // if false, non-coherent staging memory may be used and is flushed before GPU copy
    bool forceCoherentMapping = true;

    std::function<void*(VkCommandBuffer cmd, uint32_t submitCount, nvvk::SemaphoreState& sem, bool begin, void* payload)> onSubmitCallback =
        nullptr;
  };

  VkResult init(const InitInfo& info);
  void     deinit();

  void setFlushSize(uint32_t flushSize) { m_info.flushSize = flushSize; }

  uint32_t                         getSubmitCount() const { return m_submitCount; }
  uint32_t                         getWaitCmdCount() const { return m_waitCmdCount; }
  nvvk::BufferSubAllocator::Report getAllocationReport() const;
  bool                             hasOwnershipBarriers() const;

  // should be called at end of multiple upload append operations,
  // and/or once every frame
  // `semaphoreState` can be used to wait for anything up to this point
  // if there is no pending work does same as `releaseSubmitted`
  VkResult flushPending(nvvk::SemaphoreState* outSemaphoreState = nullptr);

  // waits for completion of last submit
  VkResult waitForCompletion();

  // should be called once every frame
  void releaseCompletedAllocations();

  // should be called after upload appends were done, to get a rough idea
  // what to wait for.
  // if there are pending operations left, returns next submit state
  // if flushed returns previous submit
  //
  // due to async nature when you query after an appended uploaded, you might
  // not get exactly the submit your append was part of (as another thread could trigger a flush),
  // but typically this is good enough / guaranteed conservative.
  nvvk::SemaphoreState getSubmitState();

  // apply this on destination graphics queue, prior using any resources, typically once per frame.
  void cmdDrainOwnershipBarriers(VkCommandBuffer cmd);

  // UploaderInterface
  VkResult acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle) override;
  void     releaseMapping(MappingHandle mappingHandle) override;

  VkResult appendBufferRange(const nvvk::BufferRange& buffer, const void* data, nvvk::SemaphoreState* outSemaphoreState) override;
  VkResult appendBufferRangeMappings(size_t                    rangeCount,
                                     const MappedBufferRanges* ranges,
                                     MappingHandle             mappingHandle,
                                     bool                      releaseMapping_   = true,
                                     nvvk::SemaphoreState*     outSemaphoreState = nullptr) override;

  VkResult appendImageSub(nvvk::Image&                    image,
                          const VkOffset3D&               offset,
                          const VkExtent3D&               extent,
                          const VkImageSubresourceLayers& subresource,
                          size_t                          dataSize,
                          const void*                     data,
                          VkImageLayout                   newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                          nvvk::SemaphoreState*           outSemaphoreState = nullptr) override;
  VkResult appendImageSubMappings(nvvk::Image&           image,
                                  size_t                 imageSubCount,
                                  const MappedImageSubs* imageSubs,
                                  MappingHandle          mappingHandle,
                                  bool                   releaseMapping_   = true,
                                  VkImageLayout          newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                  nvvk::SemaphoreState*  outSemaphoreState = nullptr) override;

protected:
  union MappingHandleDetail
  {
    MappingHandle handle;
    struct
    {
      uint16_t valid;
      uint16_t blockIndex;
      uint32_t mappingIndex;
    };
  };

  VkResult acquireStagingSpace(nvvk::BufferSubAllocation& subAllocation, nvvk::BufferRange& stagingSpace, size_t dataSize, const void* data);

  void addStagingAllocation(const nvvk::BufferSubAllocation& subAllocation);
  void releaseStagingAllocations(bool forceAll);

  VkResult submitBatch();
  VkResult submitBatchChecked(size_t addedSize);

  bool consumeMappingHandle(MappingHandle handle, nvvk::BufferSubAllocation& subAllocation);

  void markRangeForFlush(uint32_t blockIndex, VkDeviceSize offset, VkDeviceSize size);
  void markForFlush(const nvvk::BufferSubAllocation& subAllocation);
  void flushBatchBlocks();

  inline void setOutSemaphoreState(nvvk::SemaphoreState* out)
  {
    if(out != nullptr)
    {
      *out = m_transferSubmitState;
    }
  }

  struct StagingAllocation
  {
    nvvk::BufferSubAllocation subAllocation;
    nvvk::SemaphoreState      semaphoreState;
  };

  struct TargetBarriers
  {
    nvvk::BarrierContainer ownerAcquisitionBarriers;
    nvvk::SemaphoreState   semaphoreState;
  };

  static constexpr uint32_t MAX_FLUSH_BLOCKS = 64;
  static constexpr uint32_t RANGES_PER_BLOCK = 64;
  static constexpr uint64_t ALL_RANGE_BITS   = ~0ull;

  InitInfo               m_info;
  nvvk::StagingCopyBatch m_batch;
  // non-coherent flushing is done in ranges based on this granularity
  size_t m_flushRangeSize = 128;
  // a bit mask of which ranges we already flushed for each block
  std::array<uint64_t, MAX_FLUSH_BLOCKS> m_batchFlushRangeMasks{};

  // statistics
  // how often we had to wait for command buffer
  uint32_t m_waitCmdCount{0};
  // how many submits
  uint32_t m_submitCount{0};

  mutable std::mutex       m_allocLock;
  nvvk::BufferSubAllocator m_bufferSubAllocator;
  std::atomic_int32_t      m_allocationCount{0};

  std::mutex                             m_mappingLock;
  nvutils::IDPool                        m_mappingPool;
  std::vector<nvvk::BufferSubAllocation> m_mappingAllocations;

  std::mutex                     m_appendLock;
  std::vector<StagingAllocation> m_stagingAllocations;
  size_t                         m_stagingAllocationsSize{0};

  nvvk::SemaphoreState      m_transferSubmitState;
  nvvk::ManagedCommandPools m_transferCmdPool;
  VkSemaphore               m_transferTimelineSemaphore{};
  uint64_t                  m_transferTimelineValue = 1;

  mutable std::mutex          m_targetLock;
  std::vector<TargetBarriers> m_targetBarriers;
};
