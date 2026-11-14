/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdio>
#include <filesystem>
#include <vector>

#include <volk.h>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>

namespace lodclusters {

// Minimal H.264 encoder on a Vulkan Video encode queue. An IDR picture every `idrPeriod` frames,
// P pictures in between that reference the previous one, constant QP, written as an Annex B
// elementary stream.
//
// Each slot is a source picture the graphics queue fills (`cmdUpload`) and the encode queue reads
// (`submit`), waiting on the semaphore of the frame that filled it. `collect` waits for the encode
// and appends its bitstream. The caller collects a slot before it uploads into it again, and
// submits and collects in frame order.
class VideoEncoder
{
public:
  struct Config
  {
    VkExtent2D            size{};
    uint32_t              slotCount = 2;
    uint32_t              fps       = 60;
    uint32_t              qp        = 20;
    uint32_t              idrPeriod = 120;
    std::filesystem::path filename;
  };

  ~VideoEncoder() { deinit(); }

  // false with a message when the device cannot encode this
  bool init(VkDevice                 device,
            VkPhysicalDevice         physicalDevice,
            nvvk::ResourceAllocator& allocator,
            const nvvk::QueueInfo&   graphicsQueue,
            const nvvk::QueueInfo&   encodeQueue,
            const Config&            config);
  // waits for the outstanding encodes and writes them first, false when the file is incomplete
  bool deinit();

  bool isValid() const { return m_session != nullptr; }

  // records the copy of a tightly packed nv12 buffer of `config.size` into the slot's source picture
  void cmdUpload(VkCommandBuffer cmd, uint32_t slot, VkBuffer nv12);
  // encodes the slot's source picture once `wait` has signaled
  void submit(uint32_t slot, const VkSemaphoreSubmitInfo& wait);
  // waits for the slot's encode and appends its bitstream, false on an error
  bool collect(uint32_t slot);

  uint32_t getFramesWritten() const { return m_framesWritten; }
  uint64_t getBytesWritten() const { return m_bytesWritten; }

private:
  struct Slot
  {
    nvvk::Image     source;
    VkImageView     sourceView = nullptr;
    nvvk::Buffer    bitstream;
    VkCommandBuffer cmd       = nullptr;
    uint64_t        doneValue = 0;  // m_timeline value of its encode, 0 if none is pending
  };

  bool createSession();
  bool writeHeaders();

  VkDevice                 m_device    = nullptr;
  nvvk::ResourceAllocator* m_allocator = nullptr;
  nvvk::QueueInfo          m_encodeQueue{};
  Config                   m_config;
  VkExtent2D               m_codedExtent{};

  VkVideoEncodeH264ProfileInfoKHR m_h264Profile{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR};
  VkVideoProfileInfoKHR           m_profile{VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR};
  VkVideoProfileListInfoKHR       m_profileList{VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR};

  VkVideoSessionKHR           m_session = nullptr;
  VkVideoSessionParametersKHR m_params  = nullptr;
  std::vector<VkDeviceMemory> m_sessionMemory;
  VkQueryPool                 m_queryPool     = nullptr;
  VkCommandPool               m_cmdPool       = nullptr;
  VkSemaphore                 m_timeline      = nullptr;
  uint64_t                    m_timelineValue = 0;

  // two reconstructed pictures, the one written and the one referenced
  nvvk::Image m_dpb;
  VkImageView m_dpbView = nullptr;

  std::vector<Slot> m_slots;

  VkVideoEncodeH264CapabilitiesKHR m_h264Caps{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR};
  VkVideoEncodeCapabilitiesKHR     m_encodeCaps{VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR};
  VkVideoCapabilitiesKHR           m_caps{VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR};

  std::vector<uint8_t> m_headers;  // SPS and PPS, written before every IDR picture

  // picture state of the submitted encodes
  uint32_t m_framesSubmitted = 0;
  uint32_t m_idrPicId        = 0;
  uint32_t m_refFrameNum     = 0;
  int32_t  m_refPoc          = 0;
  bool     m_rateControlSet  = false;

  // collect order matches submit order, these tell the IDR pictures apart
  std::vector<bool> m_slotIsIdr;

  FILE*    m_file          = nullptr;
  uint32_t m_framesWritten = 0;
  uint64_t m_bytesWritten  = 0;
};

}  // namespace lodclusters
