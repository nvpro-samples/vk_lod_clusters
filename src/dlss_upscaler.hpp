/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <future>

#include <glm/glm.hpp>

#include "nvvk/gbuffers.hpp"
#include "nvvk/resource_allocator.hpp"
#include "nvvk/sampler_pool.hpp"

#include "dlss_wrapper.hpp"
#include "nvsdk_ngx_helpers_vk.h"

class DlssUpscaler
{
public:
  enum DlssBufferType
  {
    eDlssInputColor = 0,
    eDlssMotion,
    eDlssCount,
  };

  struct InitInfo
  {
    VkInstance               instance{};
    nvvk::ResourceAllocator* resourceAllocator{};
    uint32_t                 queueFamilyIndex{};
    // main thread queue, runs the warm-up feature's creation commands
    VkQueue queue{};

    // optional for UI/imgui
    nvvk::SamplerPool* samplerPool{};
    VkDescriptorPool   descriptorPool{};
  };

  DlssUpscaler()  = default;
  ~DlssUpscaler() = default;

  void init(const InitInfo& initInfo);
  void deinit();

  void deinitResources();
  void initUpscaler();
  // runs initUpscaler on a worker thread, ensureInitialized waits for it. The quality picks the
  // preset that is warmed up, each has its own network.
  void initAsync(NVSDK_NGX_PerfQuality_Value quality);
  bool isInitPending() const { return m_initFuture.valid(); }
  // completes initAsync if it finished, returns true if so
  bool pollInit();

  bool ensureInitialized(NVSDK_NGX_PerfQuality_Value quality);
  bool isInitialized() const { return !m_initFuture.valid() && m_initialized; }
  bool isAvailable() const;
  bool isActive() const;

  VkExtent2D updateSize(VkCommandBuffer cmd, VkExtent2D size, NVSDK_NGX_PerfQuality_Value quality);

  void setOutputResource(VkImage image, VkImageView imageView, VkFormat format);
  void setDepthResource(VkImage image, VkImageView imageView, VkFormat format);

  bool upscale(VkCommandBuffer cmd, glm::vec2 jitter, bool reset = false);

  VkExtent2D getRenderSize() const { return m_dlssGBuffers.getSize(); }

  const nvvk::GBuffer& getGBuffer() const { return m_dlssGBuffers; }
  nvvk::GBuffer&       getGBuffer() { return m_dlssGBuffers; }

  const std::vector<VkFormat>& getBufferFormats() const { return m_bufferInfos; }
  VkFormat                     getColorFormat(DlssBufferType type) const { return m_bufferInfos[type]; }

private:
  void warmupFeature();
  // main thread, after the init finished
  void finishWarmup(bool submit);

  InitInfo m_info{};
  bool     m_initialized  = false;
  bool     m_hasResources = false;
  bool     m_dlssCreated  = false;

  std::future<void> m_initFuture;

  NgxContext          m_ngx{};
  DlssSuperResolution m_dlss{};

  DlssSuperResolution         m_warmup{};
  VkCommandPool               m_warmupPool{};
  VkCommandBuffer             m_warmupCmd{};
  NVSDK_NGX_PerfQuality_Value m_warmupQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;

  std::vector<VkFormat> m_bufferInfos = {
      {VK_FORMAT_R16G16B16A16_SFLOAT},  // DLSS input color
      {VK_FORMAT_R16G16_SFLOAT},        // motion vectors
  };

  nvvk::GBuffer m_dlssGBuffers{};
  bool          m_dlssSupported = false;
  VkExtent2D    m_renderingSize{};
  VkDevice      m_device{};
  VkSampler     m_sampler{};
};
