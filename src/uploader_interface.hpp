/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>

// This interface will yield control over the actual submits of copying to the implementation.
// As well as the SemaphoreState management. However, each appended upload can query for
// the SempahoreState to track its completion, if required.
class UploaderInterface
{
public:
  using MappingHandle = uint64_t;

  struct MappedBufferRanges
  {
    nvvk::BufferRange bufferRange;
    nvvk::BufferRange mappingRange;
  };

  struct MappedImageSubs
  {
    VkOffset3D               offset;
    VkExtent3D               extent;
    VkImageSubresourceLayers subresource;
    nvvk::BufferRange        mappingSpace;
  };

  // This interface allows decoupling acquiring the staging space and writing to it
  // from the submit of the copy command later. In a multi-threaded scenario each
  // thread should acquire a bit of a bigger chunk that it fills and then uses for uploads.

  virtual VkResult acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle) = 0;
  virtual void     releaseMapping(MappingHandle mappingHandle)                                                    = 0;

  // if `outSemaphoreState` is provided it will store the semaphore state that can be
  // used to track completion

  inline VkResult appendBuffer(const nvvk::Buffer& buffer, size_t offset, size_t dataSize, const void* data, nvvk::SemaphoreState* outSemaphoreState = nullptr)
  {
    nvvk::BufferRange bufferRange;
    bufferRange.buffer  = buffer.buffer;
    bufferRange.offset  = offset;
    bufferRange.range   = dataSize;
    bufferRange.mapping = buffer.mapping ? buffer.mapping + offset : 0;
    bufferRange.address = buffer.address + offset;

    return appendBufferRange(bufferRange, data, outSemaphoreState);
  }

  inline VkResult appendBufferMapping(const nvvk::Buffer&      buffer,
                                      size_t                   offset,
                                      size_t                   dataSize,
                                      const nvvk::BufferRange& mappingSpace,
                                      MappingHandle            mappingHandle,
                                      bool                     releaseMapping_   = true,
                                      nvvk::SemaphoreState*    outSemaphoreState = nullptr)
  {
    MappedBufferRanges mappedRange;
    mappedRange.bufferRange.buffer  = buffer.buffer;
    mappedRange.bufferRange.offset  = offset;
    mappedRange.bufferRange.range   = dataSize;
    mappedRange.bufferRange.mapping = buffer.mapping ? buffer.mapping + offset : 0;
    mappedRange.bufferRange.address = buffer.address + offset;
    mappedRange.mappingRange        = mappingSpace;

    return appendBufferRangeMappings(1, &mappedRange, mappingHandle, releaseMapping_, outSemaphoreState);
  }

  virtual VkResult appendBufferRange(const nvvk::BufferRange& bufferRange, const void* data, nvvk::SemaphoreState* outSemaphoreState) = 0;
  virtual VkResult appendBufferRangeMappings(size_t                    rangeCount,
                                             const MappedBufferRanges* ranges,
                                             MappingHandle             mappingHandle,
                                             bool                      releaseMapping_   = true,
                                             nvvk::SemaphoreState*     outSemaphoreState = nullptr) = 0;

  virtual VkResult appendImageSub(nvvk::Image&                    image,
                                  const VkOffset3D&               offset,
                                  const VkExtent3D&               extent,
                                  const VkImageSubresourceLayers& subresource,
                                  size_t                          dataSize,
                                  const void*                     data,
                                  VkImageLayout                   newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                  nvvk::SemaphoreState*           outSemaphoreState = nullptr)         = 0;
  virtual VkResult appendImageSubMappings(nvvk::Image&           image,
                                          size_t                 imageSubCount,
                                          const MappedImageSubs* imageSubs,
                                          MappingHandle          mappingHandle,
                                          bool                   releaseMapping_   = true,
                                          VkImageLayout          newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                          nvvk::SemaphoreState*  outSemaphoreState = nullptr) = 0;
};
