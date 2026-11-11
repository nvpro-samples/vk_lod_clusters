/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <imgui/imgui.h>

#include "dlss_denoiser.hpp"

void DlssDenoiser::initAsync()
{
  if(!m_initialized && !m_initFuture.valid())
  {
    m_initFuture = std::async(std::launch::async, [this]() { initDenoiser(); });
  }
}

bool DlssDenoiser::pollInit()
{
  if(m_initFuture.valid() && m_initFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    m_initFuture.get();
    finishWarmup(true);
    return true;
  }
  return false;
}

bool DlssDenoiser::ensureInitialized()
{
  if(m_initFuture.valid())
  {
    m_initFuture.get();
    finishWarmup(true);
    return true;
  }
  if(!m_initialized)
  {
    initDenoiser();
    finishWarmup(true);
    return true;
  }
  return false;
}

void DlssDenoiser::init(const InitInfo& info)
{
  assert(m_info.resourceAllocator == nullptr);

  m_info = info;
  if(info.samplerPool)
  {
    info.samplerPool->acquireSampler(m_sampler);
  }
}

void DlssDenoiser::deinit()
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

void DlssDenoiser::deinitResources()
{
  m_dlssGBuffers.deinit();
  m_dlss.deinit();
  m_hasResources = false;
  m_dlssCreated  = false;
}

void DlssDenoiser::initDenoiser()
{
  if(m_initialized)
    return;
  SCOPED_TIMER("Initializing DLSS Denoiser");

  m_device = m_info.resourceAllocator->getDevice();

  // #DLSS - Create the DLSS
  NgxContext::InitInfo ngxInitInfo{
      .instance       = m_info.instance,
      .physicalDevice = m_info.resourceAllocator->getPhysicalDevice(),
      .device         = m_info.resourceAllocator->getDevice(),
  };
  // ngxInitInfo.loggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;

  m_dlssSupported            = false;
  NVSDK_NGX_Result ngxResult = m_ngx.init(ngxInitInfo);
  if(ngxResult == NVSDK_NGX_Result_Success)
  {
    m_dlssSupported = (m_ngx.isDlssRRAvailable() == NVSDK_NGX_Result_Success);
  }

  if(!m_dlssSupported)
  {
    LOGW("NGX init failed: %d - DLSS unsupported\n", ngxResult);
  }
  else
  {
    warmupFeature();
  }
  m_initialized = true;
}

void DlssDenoiser::warmupFeature()
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

  DlssRayReconstruction::SupportedSizes supportedSizes{};
  if(NVSDK_NGX_SUCCEED(DlssRayReconstruction::querySupportedInputSizes(m_ngx, {{128, 128}, NVSDK_NGX_PerfQuality_Value_MaxQuality},
                                                                       &supportedSizes)))
  {
    m_warmup.cmdInit(m_warmupCmd, m_ngx,
                     {.depthInverted = true, .inputSize = supportedSizes.optimalSize, .outputSize = {128, 128}});
  }

  NVVK_CHECK(vkEndCommandBuffer(m_warmupCmd));
}

void DlssDenoiser::finishWarmup(bool submit)
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


VkDescriptorImageInfo DlssDenoiser::getDescriptorImageInfo(DlssBufferType name) const
{
  assert(m_hasResources);
  return m_dlssGBuffers.getDescriptorImageInfo(name);
}

bool DlssDenoiser::isAvailable() const
{
  return !m_initFuture.valid() && m_dlssSupported && m_initialized;
}

bool DlssDenoiser::isActive() const
{
  return isAvailable() && m_hasResources && m_dlssCreated;
}

VkExtent2D DlssDenoiser::updateSize(VkCommandBuffer cmd, VkExtent2D size, NVSDK_NGX_PerfQuality_Value quality)
{
  if(!isAvailable())
    return size;

  m_dlssCreated = false;

  // Choose the size of the DLSS buffers
  DlssRayReconstruction::SupportedSizes supportedSizes{};
  NVSDK_NGX_Result ngxResult = DlssRayReconstruction::querySupportedInputSizes(m_ngx, {size, quality}, &supportedSizes);
  // unsupported quality modes report success with a zero size
  if(NVSDK_NGX_FAILED(ngxResult) || supportedSizes.optimalSize.width == 0 || supportedSizes.optimalSize.height == 0)
  {
    LOGE("DLSS-RR size query failed: %d\n", ngxResult);
    return size;
  }
  m_renderingSize = supportedSizes.optimalSize;

  DlssRayReconstruction::InitInfo initInfo{
      .quality       = quality,
      .depthInverted = true,
      .inputSize     = m_renderingSize,
      .outputSize    = size,
  };
  m_dlss.deinit();
  vkDeviceWaitIdle(m_device);
  ngxResult = m_dlss.cmdInit(cmd, m_ngx, initInfo);
  if(NVSDK_NGX_FAILED(ngxResult))
  {
    LOGE("DLSS-RR feature creation failed: %d\n", ngxResult);
    return size;
  }
  LOGI("DLSS-RR feature created: input=%ux%u output=%ux%u quality=%d\n", m_renderingSize.width, m_renderingSize.height,
       size.width, size.height, int(quality));

  if(!m_hasResources)
  {
    // G-Buffer
    m_dlssGBuffers.init({.allocator      = m_info.resourceAllocator,
                         .colorFormats   = m_bufferInfos,
                         .imageSampler   = m_sampler,
                         .descriptorPool = m_info.descriptorPool});
  }

  // Recreate the G-Buffers
  m_dlssGBuffers.update(cmd, m_renderingSize);
  m_hasResources = true;

  auto dlssResourceFromGBufTexture = [&](DlssRayReconstruction::ResourceType resource, DlssBufferType gbufIndex) {
    m_dlss.setResource({resource, m_dlssGBuffers.getColorImage(gbufIndex), m_dlssGBuffers.getColorImageView(gbufIndex),
                        m_dlssGBuffers.getColorFormat(gbufIndex)});
  };

  // #DLSS Fill the user pool with our textures
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eColorIn, DlssBufferType::eDlssRenderImage);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eDiffuseAlbedo, DlssBufferType::eDlssAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eSpecularAlbedo, DlssBufferType::eDlssSpecAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eNormalRoughness, DlssBufferType::eDlssNormalRoughness);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eMotionVector, DlssBufferType::eDlssMotion);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eSpecularHitDistance, DlssBufferType::eDlssSpecHitDist);

  m_dlssCreated = true;

  return m_renderingSize;
}

void DlssDenoiser::setResource(DlssRayReconstruction::ResourceType resourceId, VkImage image, VkImageView imageView, VkFormat format)
{
  m_dlss.setResource({resourceId, image, imageView, format});
}

bool DlssDenoiser::denoise(VkCommandBuffer cmd, glm::vec2 jitter, const glm::mat4& modelView, const glm::mat4& projection, bool reset /*= false*/)
{
  if(!isActive())
    return false;

  NVSDK_NGX_Result ngxResult = m_dlss.cmdDenoise(cmd, m_ngx, {jitter, modelView, projection, reset});
  if(NVSDK_NGX_FAILED(ngxResult))
  {
    LOGE("DLSS-RR evaluation failed: %d\n", ngxResult);
    return false;
  }

  static bool s_loggedFirstEvaluation = false;
  if(!s_loggedFirstEvaluation)
  {
    LOGI("DLSS-RR first evaluation succeeded.\n");
    s_loggedFirstEvaluation = true;
  }
  return true;
}

void DlssDenoiser::onUi()
{
  if(!isActive())
  {
    return;
  }

  ImVec2 tumbnailSize = {100 * m_dlssGBuffers.getAspectRatio(), 100};
  int    m_showBuffer = -1;
  auto   showBuffer   = [&](const char* name, DlssBufferType buffer) {
    ImGui::Text("%s", name);
    if(ImGui::ImageButton(name, ImTextureID(m_dlssGBuffers.getDescriptorSet(buffer)), tumbnailSize))
      m_showBuffer = buffer;
  };

  if(ImGui::CollapsingHeader("DLSS Guide Images"))
  {
    if(ImGui::BeginTable("thumbnails", 2))
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      showBuffer("Color", DlssBufferType::eDlssAlbedo);
      ImGui::TableNextColumn();
      showBuffer("Normal", DlssBufferType::eDlssNormalRoughness);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      showBuffer("Motion", DlssBufferType::eDlssMotion);
      ImGui::TableNextColumn();
      showBuffer("Specular Albedo", DlssBufferType::eDlssSpecAlbedo);
      ImGui::EndTable();
    }
  }
}
