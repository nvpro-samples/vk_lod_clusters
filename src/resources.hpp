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

#include <algorithm>
#include <cassert>

#include <platform.h>
#include <nvh/nvprint.hpp>
#include <nvh/alignment.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvkhl/sky.hpp>

#include "hbao_pass.hpp"
#include "nvhiz_vk.hpp"
#include "shaders/shaderio.h"

namespace lodclusters {

struct FrameConfig
{
  uint32_t winWidth;
  uint32_t winHeight;

  bool  showInstanceBboxes = false;
  bool  freezeCulling      = false;
  bool  hbaoActive         = true;
  float lodPixelError      = 1.0f;
  // how many frames until we schedule a group for unloading
  uint32_t streamingAgeThreshold = 16;
  // how much threads to use in the persistent kernels
  uint32_t traversalPersistentThreads = 2048;

  HbaoPass::Settings hbaoSettings;

  // must be kept next to each other
  shaderio::FrameConstants frameConstants;
  shaderio::FrameConstants frameConstantsLast;
};

//////////////////////////////////////////////////////////////////////////

struct RBuffer : nvvk::Buffer
{
  VkDescriptorBufferInfo info    = {VK_NULL_HANDLE};
  void*                  mapping = nullptr;
};

// allows > 4 GB allocations using sparse memory
struct RLargeBuffer : nvvk::LargeBuffer
{
  VkDescriptorBufferInfo info = {VK_NULL_HANDLE};
};

template <class T>
struct RBufferTyped : RBuffer
{
  typedef T value_type;

  RBufferTyped& operator=(RBuffer other)
  {
    *(RBuffer*)this = other;
    return *this;
  }

  size_t   size() const { return info.range / sizeof(T); }
  const T* data() const { return static_cast<const T*>(mapping); }
  T*       data() { return static_cast<T*>(mapping); }

  VkDeviceAddress addressAt(size_t idx) const
  {
    assert(idx < size());
    return address + sizeof(T) * idx;
  }

  VkDeviceAddress addressRange(size_t start, size_t num) const
  {
    assert(start + num <= size());
    return address + sizeof(T) * start;
  }
};

struct RImage : nvvk::Image
{
  RImage() {}
  RImage& operator=(nvvk::Image other)
  {
    *(nvvk::Image*)this = other;
    layout              = VK_IMAGE_LAYOUT_UNDEFINED;
    return *this;
  }

  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

inline void cmdCopyBuffer(VkCommandBuffer cmd, const RBuffer& src, const RBuffer& dst)
{
  VkBufferCopy cpy = {src.info.offset, dst.info.offset, src.info.range};
  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}

//////////////////////////////////////////////////////////////////////////

#define DEBUGUTIL_SET_NAME(var) debugUtil.setObjectName(var, #var)

struct BufferRanges
{
  VkDeviceSize tempOffset = 0;

  VkDeviceSize beginOffset = 0;
  VkDeviceSize splitOffset = 0;

  VkDeviceSize append(VkDeviceSize size, VkDeviceSize alignment)
  {
    tempOffset = nvh::align_up(tempOffset, alignment);

    VkDeviceSize offset = tempOffset;
    tempOffset += size;

    return offset;
  }

  void beginOverlap()
  {
    beginOffset = tempOffset;
    splitOffset = 0;
  }
  void splitOverlap()
  {
    splitOffset = std::max(splitOffset, tempOffset);
    tempOffset  = beginOffset;
  }
  void endOverlap() { tempOffset = std::max(splitOffset, tempOffset); }

  VkDeviceSize getSize(VkDeviceSize alignment = 4) { return nvh::align_up(tempOffset, alignment); }
};

//////////////////////////////////////////////////////////////////////////

struct SemaphoreState
{
  VkSemaphore semaphore = nullptr;
  uint64_t    value     = 0;

  bool wait(VkDevice device, uint64_t timeout) const
  {
    VkSemaphoreWaitInfo waitInfo{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &semaphore,
        .pValues        = &value,
    };
    VkResult r = vkWaitSemaphores(device, &waitInfo, timeout);
    if(r == VK_TIMEOUT)
      return false;
    return true;
  }

  bool isAvailable(VkDevice device) const
  {
    uint64_t currentValue;
    VkResult r = vkGetSemaphoreCounterValue(device, semaphore, &currentValue);
    if(r == VK_SUCCESS)
      return currentValue >= value;
    return false;
  }
};

class QueueState
{
public:
  VkDevice    m_device            = nullptr;
  VkQueue     m_queue             = nullptr;
  uint32_t    m_familyIndex       = 0;
  VkSemaphore m_timelineSemaphore = nullptr;
  uint64_t    m_timelineValue     = 1;

  std::vector<VkSemaphoreSubmitInfo> m_pendingWaits;

  void init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue);
  void deinit();

  VkResult getTimelineValue(uint64_t& timelineValue) const
  {
    return vkGetSemaphoreCounterValue(m_device, m_timelineSemaphore, &timelineValue);
  }

  SemaphoreState getCurrentState() const { return {m_timelineSemaphore, m_timelineValue}; }

  VkSemaphoreSubmitInfo getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0) const;

  // increments timeline
  VkSemaphoreSubmitInfo advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0);
};

struct QueueStateManager
{
  QueueState primary;
  QueueState transfer;
};

//////////////////////////////////////////////////////////////////////////

class Resources
{
public:
  struct FrameBuffer
  {
    int  renderWidth  = 0;
    int  renderHeight = 0;
    int  supersample  = 0;
    bool useResolved  = false;

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthStencilFormat;

    VkViewport viewport;
    VkRect2D   scissor;

    RImage imgColor         = {};
    RImage imgColorResolved = {};
    RImage imgDepthStencil  = {};

    VkImageView viewColor         = VK_NULL_HANDLE;
    VkImageView viewColorResolved = VK_NULL_HANDLE;
    VkImageView viewDepthStencil  = VK_NULL_HANDLE;
    VkImageView viewDepth         = VK_NULL_HANDLE;

    VkFormat    raytracingDepthFormat = VK_FORMAT_R32_SFLOAT;
    RImage      imgRaytracingDepth    = {};
    VkImageView viewRaytracingDepth   = VK_NULL_HANDLE;

    RImage imgHizFar = {};

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  };

  struct CommonResources
  {
    RBuffer view;
    RBuffer readbackDevice;
    RBuffer readbackHost;
  };

  const nvvk::Context* m_context;
  VkDevice             m_device = VK_NULL_HANDLE;
  VkPhysicalDevice     m_physical;
  VkQueue              m_queue;
  uint32_t             m_queueFamily;

  VkPipelineStageFlags m_supportedSaderStageFlags;

  nvvk::DeviceMemoryAllocator m_memAllocator;
  nvvk::ResourceAllocator     m_allocator;

  NVHizVK              m_hiz;
  NVHizVK::Update      m_hizUpdate;
  nvvk::ShaderModuleID m_hizShaders[NVHizVK::SHADER_COUNT];

  bool            m_hbaoFullRes = false;
  HbaoPass        m_hbaoPass;
  HbaoPass::Frame m_hbaoFrame;

  nvvk::CommandPool           m_tempCommandPool;
  nvvk::ShaderModuleManager   m_shaderManager;
  nvvk::GraphicsPipelineState m_basicGraphicsState;

  nvvkhl::SimpleSkyDome m_sky;

  CommonResources   m_common;
  FrameBuffer       m_framebuffer;
  QueueStateManager m_queueStates;

  uint32_t m_cycleIndex  = 0;
  size_t   m_fboChangeID = ~0;

  glm::vec4 m_bgColor = {0.1, 0.13, 0.15, 1.0};

  bool init(nvvk::Context* context, const std::vector<std::string>& shaderSearchPaths);
  void deinit();

  bool initFramebuffer(int width, int height, int supersample, bool hbaoFullRes);
  void deinitFramebuffer();

  void synchronize(const char* debugMsg = nullptr);

  void beginFrame(uint32_t cycleIndex);
  void postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler);
  void emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler);
  void endFrame();

  void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler);
  void cmdHBAO(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler);

  void getReadbackData(shaderio::Readback& readback);

  //////////////////////////////////////////////////////////////////////////

  // tests if all shaders compiled well, returns false if not
  // also destroys all shaders if not all were successful.
  bool verifyShaders(size_t numShaders, nvvk::ShaderModuleID* shaders);
  template <typename T>
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(nvvk::ShaderModuleID), (nvvk::ShaderModuleID*)&container);
  }

  void destroyShaders(size_t numShaders, nvvk::ShaderModuleID* shaders);
  template <typename T>
  void destroyShaders(T& container)
  {
    destroyShaders(sizeof(T) / sizeof(nvvk::ShaderModuleID), (nvvk::ShaderModuleID*)&container);
  }

  //////////////////////////////////////////////////////////////////////////

  VkDeviceSize getDeviceLocalHeapSize() const;
  bool         isBufferSizeValid(VkDeviceSize size) const;

  RBuffer createBuffer(VkDeviceSize                 size,
                       VkBufferUsageFlags           flags,
                       VkMemoryPropertyFlags        memFlags             = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       const std::vector<uint32_t>* sharingQueueFamilies = nullptr);

  template <class T>
  void createBufferTyped(RBufferTyped<T>&             rbufferTyped,
                         size_t                       elementCount,
                         VkBufferUsageFlags           flags,
                         VkMemoryPropertyFlags        memFlags             = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         const std::vector<uint32_t>* sharingQueueFamilies = nullptr)
  {
    rbufferTyped = createBuffer(sizeof(T) * elementCount, flags, memFlags, sharingQueueFamilies);
  }

  RLargeBuffer createLargeBuffer(VkDeviceSize                 size,
                                 VkBufferUsageFlags           flags,
                                 VkMemoryPropertyFlags        memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 const std::vector<uint32_t>* sharingQueueFamilies = nullptr);

  void destroy(RBuffer& obj);
  void destroy(RLargeBuffer& obj);

  nvvk::AccelKHR createAccelKHR(VkAccelerationStructureCreateInfoKHR& createInfo);
  void           destroy(nvvk::AccelKHR& obj);

  //////////////////////////////////////////////////////////////////////////

  void simpleUploadBuffer(const RBuffer& dst, const void* src);
  void simpleUploadBuffer(const RBuffer& dst, size_t offset, size_t sz, const void* src);
  void simpleDownloadBuffer(void* dst, const RBuffer& src);

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createTempCmdBuffer();
  void            tempSyncSubmit(VkCommandBuffer cmd, bool reset = true);
  void            tempAbort(VkCommandBuffer cmd, bool reset = true);
  void            tempResetResources();

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear, bool isCompute = false) const;

  //////////////////////////////////////////////////////////////////////////

  void cmdBeginRendering(VkCommandBuffer    cmd,
                         bool               hasSecondary = false,
                         VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR);
  void cmdDynamicState(VkCommandBuffer cmd) const;
  void cmdBegin(VkCommandBuffer cmd, bool singleshot, bool primary, bool secondaryInClear, bool isCompute = false) const;

  void cmdImageTransition(VkCommandBuffer cmd, RImage& img, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;

  //////////////////////////////////////////////////////////////////////////

  enum FlushState
  {
    ALLOW_FLUSH,
    DONT_FLUSH,
  };

  class BatchedUploader
  {
  public:
    BatchedUploader(Resources& resources, VkDeviceSize maxBatchSize = 128 * 1024 * 1024)
        : m_resources(resources)
        , m_maxBatchSize(maxBatchSize)
    {
    }

    VkCommandBuffer getCmd()
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      return m_cmd;
    }

    template <class T>
    T* uploadBuffer(const RBuffer& dst, size_t offset, size_t sz, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      if(sz)
      {
        if(m_batchSize && m_batchSize + sz > m_maxBatchSize && flushState == FlushState::ALLOW_FLUSH)
        {
          flush();
        }

        m_batchSize += sz;
        return static_cast<T*>(m_resources.m_allocator.getStaging()->cmdToBuffer(getCmd(), dst.buffer, offset, sz, src));
      }
      return nullptr;
    }
    template <class T>
    T* uploadBuffer(const RBuffer& dst, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      return uploadBuffer(dst, 0, dst.info.range, src, flushState);
    }

    void fillBuffer(const RBuffer& dst, uint32_t fillValue)
    {
      vkCmdFillBuffer(getCmd(), dst.buffer, 0, dst.info.range, fillValue);
    }

    // must call flush at end of operations
    void flush()
    {
      if(m_cmd)
      {
        m_resources.tempSyncSubmit(m_cmd);
        m_cmd       = nullptr;
        m_batchSize = 0;
      }
    }

    void abort()
    {
      if(m_cmd)
      {
        m_resources.tempAbort(m_cmd);
        m_cmd       = nullptr;
        m_batchSize = 0;
      }
    }

    ~BatchedUploader() { assert(!m_cmd); }

  private:
    Resources&      m_resources;
    VkDeviceSize    m_maxBatchSize = 0;
    VkDeviceSize    m_batchSize    = 0;
    VkCommandBuffer m_cmd          = nullptr;
  };
};


}  // namespace lodclusters
