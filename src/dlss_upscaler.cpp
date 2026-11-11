/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "dlss_upscaler.hpp"

void DlssUpscaler::initAsync(NVSDK_NGX_PerfQuality_Value quality)
{
  if(!m_initialized && !m_initFuture.valid())
  {
    m_warmupQuality = quality;
    m_initFuture    = std::async(std::launch::async, [this]() { initUpscaler(); });
  }
}

bool DlssUpscaler::pollInit()
{
  if(m_initFuture.valid() && m_initFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    m_initFuture.get();
    finishWarmup(true);
    return true;
  }
  return false;
}

bool DlssUpscaler::ensureInitialized(NVSDK_NGX_PerfQuality_Value quality)
{
  if(m_initFuture.valid())
  {
    m_initFuture.get();
    finishWarmup(true);
    return true;
  }
  if(!m_initialized)
  {
    m_warmupQuality = quality;
    initUpscaler();
    finishWarmup(true);
    return true;
  }
  return false;
}

void DlssUpscaler::init(const InitInfo& info)
{
  assert(m_info.resourceAllocator == nullptr);

  m_info = info;
  if(info.samplerPool)
  {
    info.samplerPool->acquireSampler(m_sampler);
  }
}

void DlssUpscaler::deinit()
{
  if(m_initFuture.valid())
  {
    m_initFuture.get();
  }
  finishWarmup(false);
  if(m_info.resourceAllocator != nullptr)
  {
    m_dlssGBuffers.deinit();
    m_dlss.deinit();
    m_ngx.deinit();
    m_initialized  = false;
    m_hasResources = false;
    m_dlssCreated  = false;
    if(m_info.samplerPool)
    {
      //m_info.samplerPool->releaseSampler(m_sampler);
    }

    m_info = {};
  }
}

void DlssUpscaler::deinitResources()
{
  m_dlssGBuffers.deinit();
  m_dlss.deinit();
  m_hasResources = false;
  m_dlssCreated  = false;
}

void DlssUpscaler::initUpscaler()
{
  if(m_initialized)
    return;
  SCOPED_TIMER("Initializing DLSS Upscaler");

  m_device = m_info.resourceAllocator->getDevice();

  NgxContext::InitInfo ngxInitInfo{
      .instance       = m_info.instance,
      .physicalDevice = m_info.resourceAllocator->getPhysicalDevice(),
      .device         = m_info.resourceAllocator->getDevice(),
  };

  m_dlssSupported            = false;
  NVSDK_NGX_Result ngxResult = m_ngx.init(ngxInitInfo);
  if(ngxResult == NVSDK_NGX_Result_Success)
  {
    m_dlssSupported = (m_ngx.isDlssSRAvailable() == NVSDK_NGX_Result_Success);
  }

  if(!m_dlssSupported)
  {
    LOGW("NGX init failed: %d - DLSS-SR unsupported\n", ngxResult);
  }
  else
  {
    warmupFeature();
  }
  m_initialized = true;
}

void DlssUpscaler::warmupFeature()
{
  // the first feature creation is slow, do it once with a throw-away feature. Its commands initialize
  // state that later features share, finishWarmup submits them.
  const VkCommandPoolCreateInfo poolInfo{.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                         .queueFamilyIndex = m_info.queueFamilyIndex};
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_warmupPool));
  const VkCommandBufferAllocateInfo allocInfo{.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                              .commandPool        = m_warmupPool,
                                              .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                              .commandBufferCount = 1};
  NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &m_warmupCmd));
  const VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  NVVK_CHECK(vkBeginCommandBuffer(m_warmupCmd, &beginInfo));

  DlssSuperResolution::SupportedSizes supportedSizes{};
  if(NVSDK_NGX_SUCCEED(DlssSuperResolution::querySupportedInputSizes(m_ngx, {{128, 128}, m_warmupQuality}, &supportedSizes)))
  {
    m_warmup.cmdInit(m_warmupCmd, m_ngx,
                     {.quality = m_warmupQuality, .inputSize = supportedSizes.optimalSize, .outputSize = {128, 128}, .depthInverted = true});
  }

  NVVK_CHECK(vkEndCommandBuffer(m_warmupCmd));
}

void DlssUpscaler::finishWarmup(bool submit)
{
  if(!m_warmupPool)
    return;

  if(submit)
  {
    const VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &m_warmupCmd};
    NVVK_CHECK(vkQueueSubmit(m_info.queue, 1, &submitInfo, VK_NULL_HANDLE));
    NVVK_CHECK(vkQueueWaitIdle(m_info.queue));
  }

  m_warmup.deinit();
  vkDestroyCommandPool(m_device, m_warmupPool, nullptr);
  m_warmupPool = VK_NULL_HANDLE;
  m_warmupCmd  = VK_NULL_HANDLE;
}

bool DlssUpscaler::isAvailable() const
{
  return !m_initFuture.valid() && m_dlssSupported && m_initialized;
}

bool DlssUpscaler::isActive() const
{
  return isAvailable() && m_hasResources && m_dlssCreated;
}

VkExtent2D DlssUpscaler::updateSize(VkCommandBuffer cmd, VkExtent2D size, NVSDK_NGX_PerfQuality_Value quality)
{
  if(!isAvailable())
    return size;

  m_dlssCreated = false;

  DlssSuperResolution::SupportedSizes supportedSizes{};
  NVSDK_NGX_Result result = DlssSuperResolution::querySupportedInputSizes(m_ngx, {size, quality}, &supportedSizes);
  // unsupported quality modes report success with a zero size
  if(NVSDK_NGX_FAILED(result) || supportedSizes.optimalSize.width == 0 || supportedSizes.optimalSize.height == 0)
  {
    LOGE("DLSS_SR: Failed to query supported input sizes: %d\n", result);
    return size;
  }
  m_renderingSize = supportedSizes.optimalSize;

  DlssSuperResolution::InitInfo initInfo{
      .quality       = quality,
      .inputSize     = m_renderingSize,
      .outputSize    = size,
      .depthInverted = true,
  };
  m_dlss.deinit();
  vkDeviceWaitIdle(m_device);
  result = m_dlss.cmdInit(cmd, m_ngx, initInfo);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGE("DLSS_SR: Failed to create DLSS feature: %d\n", result);
    return size;
  }
  m_dlssCreated = true;

  if(!m_hasResources)
  {
    m_dlssGBuffers.init({.allocator      = m_info.resourceAllocator,
                         .colorFormats   = m_bufferInfos,
                         .imageSampler   = m_sampler,
                         .descriptorPool = m_info.descriptorPool});
  }

  m_dlssGBuffers.update(cmd, m_renderingSize);
  m_hasResources = true;

  auto dlssResourceFromGBufTexture = [&](DlssSuperResolution::ResourceType resource, DlssBufferType gbufIndex) {
    m_dlss.setResource({resource, m_dlssGBuffers.getColorImage(gbufIndex), m_dlssGBuffers.getColorImageView(gbufIndex),
                        m_dlssGBuffers.getColorFormat(gbufIndex)});
  };

  dlssResourceFromGBufTexture(DlssSuperResolution::ResourceType::eColorIn, DlssBufferType::eDlssInputColor);
  dlssResourceFromGBufTexture(DlssSuperResolution::ResourceType::eMotionVector, DlssBufferType::eDlssMotion);

  return m_renderingSize;
}

void DlssUpscaler::setOutputResource(VkImage image, VkImageView imageView, VkFormat format)
{
  m_dlss.setResource({DlssSuperResolution::ResourceType::eColorOut, image, imageView, format});
}

void DlssUpscaler::setDepthResource(VkImage image, VkImageView imageView, VkFormat format)
{
  m_dlss.setResource(
      {DlssSuperResolution::ResourceType::eDepth, image, imageView, format, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}});
}

bool DlssUpscaler::upscale(VkCommandBuffer cmd, glm::vec2 jitter, bool reset)
{
  if(!isActive())
    return false;

  return m_dlss.cmdUpscale(cmd, m_ngx, {jitter, reset}) == NVSDK_NGX_Result_Success;
}
