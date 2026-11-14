/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>

#include "video_encoder.hpp"

namespace lodclusters {

static constexpr VkFormat NV12_FORMAT   = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
static constexpr uint32_t DPB_SLOTS     = 2;
static constexpr uint32_t MAX_FRAME_NUM = 256;  // log2_max_frame_num_minus4 = 4
static constexpr uint8_t  NO_REFERENCE  = STD_VIDEO_H264_NO_REFERENCE_PICTURE;
static constexpr uint32_t QUERY_RESULTS = 3;  // offset, bytes written, status

static uint32_t alignUp(uint32_t value, uint32_t alignment)
{
  return (value + alignment - 1) / alignment * alignment;
}

static bool supportsNv12(VkPhysicalDevice physicalDevice, const VkVideoProfileListInfoKHR& profileList, VkImageUsageFlags usage)
{
  VkPhysicalDeviceVideoFormatInfoKHR info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR};
  info.pNext      = &profileList;
  info.imageUsage = usage;

  uint32_t count = 0;
  if(vkGetPhysicalDeviceVideoFormatPropertiesKHR(physicalDevice, &info, &count, nullptr) != VK_SUCCESS)
    return false;
  std::vector<VkVideoFormatPropertiesKHR> properties(count, {VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR});
  if(vkGetPhysicalDeviceVideoFormatPropertiesKHR(physicalDevice, &info, &count, properties.data()) != VK_SUCCESS)
    return false;

  for(const VkVideoFormatPropertiesKHR& property : properties)
  {
    if(property.format == NV12_FORMAT && property.imageTiling == VK_IMAGE_TILING_OPTIMAL)
      return true;
  }
  return false;
}

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits)
{
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
  for(uint32_t i = 0; i < properties.memoryTypeCount; i++)
  {
    if((typeBits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      return i;
  }
  for(uint32_t i = 0; i < properties.memoryTypeCount; i++)
  {
    if(typeBits & (1u << i))
      return i;
  }
  return ~0u;
}

bool VideoEncoder::init(VkDevice                 device,
                        VkPhysicalDevice         physicalDevice,
                        nvvk::ResourceAllocator& allocator,
                        const nvvk::QueueInfo&   graphicsQueue,
                        const nvvk::QueueInfo&   encodeQueue,
                        const Config&            config)
{
  assert(!m_session);
  m_device      = device;
  m_allocator   = &allocator;
  m_encodeQueue = encodeQueue;
  m_config      = config;

  m_h264Profile.stdProfileIdc   = STD_VIDEO_H264_PROFILE_IDC_HIGH;
  m_profile.pNext               = &m_h264Profile;
  m_profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
  m_profile.chromaSubsampling   = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
  m_profile.lumaBitDepth        = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
  m_profile.chromaBitDepth      = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
  m_profileList.profileCount    = 1;
  m_profileList.pProfiles       = &m_profile;

  m_encodeCaps.pNext = &m_h264Caps;
  m_caps.pNext       = &m_encodeCaps;
  VkResult result    = vkGetPhysicalDeviceVideoCapabilitiesKHR(physicalDevice, &m_profile, &m_caps);
  if(result != VK_SUCCESS)
  {
    LOGE("Video encoder: H.264 High profile encoding is not supported (%d)\n", result);
    return false;
  }

  // whole macroblocks, the SPS crops the padding
  uint32_t alignX = std::max({16u, m_caps.pictureAccessGranularity.width, m_encodeCaps.encodeInputPictureGranularity.width});
  uint32_t alignY = std::max({16u, m_caps.pictureAccessGranularity.height, m_encodeCaps.encodeInputPictureGranularity.height});
  m_codedExtent = {alignUp(config.size.width, alignX), alignUp(config.size.height, alignY)};

  if(config.size.width % 2 || config.size.height % 2 || m_codedExtent.width > m_caps.maxCodedExtent.width
     || m_codedExtent.height > m_caps.maxCodedExtent.height || m_codedExtent.width < m_caps.minCodedExtent.width
     || m_codedExtent.height < m_caps.minCodedExtent.height)
  {
    LOGE("Video encoder: %ux%u is not supported, H.264 encoding takes even sizes from %ux%u to %ux%u\n",
         config.size.width, config.size.height, m_caps.minCodedExtent.width, m_caps.minCodedExtent.height,
         m_caps.maxCodedExtent.width, m_caps.maxCodedExtent.height);
    return false;
  }

  VkVideoEncodeFeedbackFlagsKHR feedback =
      VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR | VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR;
  if(m_caps.maxDpbSlots < DPB_SLOTS || m_caps.maxActiveReferencePictures < 1 || m_h264Caps.maxPPictureL0ReferenceCount < 1
     || !(m_encodeCaps.rateControlModes & VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR)
     || (m_encodeCaps.supportedEncodeFeedbackFlags & feedback) != feedback)
  {
    LOGE("Video encoder: the device does not support P pictures with constant QP\n");
    return false;
  }
  m_config.qp = uint32_t(std::clamp(int32_t(config.qp), m_h264Caps.minQp, m_h264Caps.maxQp));

  // collect reads the encode status with the feedback
  {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyQueryResultStatusPropertiesKHR> status(count, {VK_STRUCTURE_TYPE_QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR});
    std::vector<VkQueueFamilyProperties2> families(count, {VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2});
    for(uint32_t i = 0; i < count; i++)
    {
      families[i].pNext = &status[i];
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &count, families.data());
    if(encodeQueue.familyIndex >= count || !status[encodeQueue.familyIndex].queryResultStatusSupport)
    {
      LOGE("Video encoder: the encode queue does not report query result status\n");
      return false;
    }
  }

  if(!supportsNv12(physicalDevice, m_profileList, VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
     || !supportsNv12(physicalDevice, m_profileList, VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR))
  {
    LOGE("Video encoder: nv12 pictures are not supported\n");
    return false;
  }

  if(!createSession())
  {
    deinit();
    return false;
  }

  // session memory
  {
    uint32_t count = 0;
    NVVK_CHECK(vkGetVideoSessionMemoryRequirementsKHR(m_device, m_session, &count, nullptr));
    std::vector<VkVideoSessionMemoryRequirementsKHR> requirements(count, {VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR});
    NVVK_CHECK(vkGetVideoSessionMemoryRequirementsKHR(m_device, m_session, &count, requirements.data()));

    std::vector<VkBindVideoSessionMemoryInfoKHR> binds(count, {VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR});
    for(uint32_t i = 0; i < count; i++)
    {
      VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocInfo.allocationSize  = requirements[i].memoryRequirements.size;
      allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, requirements[i].memoryRequirements.memoryTypeBits);
      VkDeviceMemory memory     = nullptr;
      NVVK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &memory));
      m_sessionMemory.push_back(memory);

      binds[i].memoryBindIndex = requirements[i].memoryBindIndex;
      binds[i].memory          = memory;
      binds[i].memorySize      = allocInfo.allocationSize;
    }
    NVVK_CHECK(vkBindVideoSessionMemoryKHR(m_device, m_session, count, binds.data()));
  }

  if(!writeHeaders())
  {
    deinit();
    return false;
  }

  // pictures, the graphics queue fills the sources
  uint32_t families[2] = {graphicsQueue.familyIndex, encodeQueue.familyIndex};
  bool     shareSource = families[0] != families[1];
  m_slots.resize(config.slotCount);
  m_slotIsIdr.resize(config.slotCount);

  VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imageInfo.pNext       = &m_profileList;
  imageInfo.imageType   = VK_IMAGE_TYPE_2D;
  imageInfo.format      = NV12_FORMAT;
  imageInfo.extent      = {m_codedExtent.width, m_codedExtent.height, 1};
  imageInfo.mipLevels   = 1;
  imageInfo.arrayLayers = DPB_SLOTS;
  imageInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage       = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR;
  NVVK_CHECK(m_allocator->createImage(m_dpb, imageInfo));

  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image            = m_dpb.image;
  viewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  viewInfo.format           = NV12_FORMAT;
  viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, DPB_SLOTS};
  NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_dpbView));

  imageInfo.arrayLayers           = 1;
  imageInfo.usage                 = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.sharingMode           = shareSource ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.queueFamilyIndexCount = shareSource ? 2 : 0;
  imageInfo.pQueueFamilyIndices   = families;
  viewInfo.viewType               = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.subresourceRange       = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  // a 4:2:0 picture compresses to well below this
  VkDeviceSize bitstreamSize = VkDeviceSize(m_codedExtent.width) * m_codedExtent.height * 3 / 2;
  bitstreamSize = (bitstreamSize + m_caps.minBitstreamBufferSizeAlignment - 1) / m_caps.minBitstreamBufferSizeAlignment
                  * m_caps.minBitstreamBufferSizeAlignment;
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.pNext = &m_profileList;
  bufferInfo.size  = bitstreamSize;
  bufferInfo.usage = VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR;
  VmaAllocationCreateInfo bufferAlloc{};
  bufferAlloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  bufferAlloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = encodeQueue.familyIndex;
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool));

  for(Slot& slot : m_slots)
  {
    NVVK_CHECK(m_allocator->createImage(slot.source, imageInfo));
    viewInfo.image = slot.source.image;
    NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &slot.sourceView));

    VmaAllocationInfo allocOut{};
    NVVK_CHECK(vmaCreateBuffer(*m_allocator, &bufferInfo, &bufferAlloc, &slot.bitstream.buffer, &slot.bitstream.allocation, &allocOut));
    slot.bitstream.bufferSize = bitstreamSize;
    slot.bitstream.mapping    = static_cast<uint8_t*>(allocOut.pMappedData);

    VkCommandBufferAllocateInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdInfo.commandPool        = m_cmdPool;
    cmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    NVVK_CHECK(vkAllocateCommandBuffers(m_device, &cmdInfo, &slot.cmd));
  }

  VkQueryPoolVideoEncodeFeedbackCreateInfoKHR feedbackInfo{VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR};
  feedbackInfo.pNext               = &m_profile;
  feedbackInfo.encodeFeedbackFlags = feedback;
  VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  queryInfo.pNext      = &feedbackInfo;
  queryInfo.queryType  = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR;
  queryInfo.queryCount = config.slotCount;
  NVVK_CHECK(vkCreateQueryPool(m_device, &queryInfo, nullptr, &m_queryPool));

  VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphoreInfo.pNext = &timelineInfo;
  NVVK_CHECK(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_timeline));

#ifdef _WIN32
  if(_wfopen_s(&m_file, config.filename.c_str(), L"wb") != 0)
    m_file = nullptr;
#else
  m_file = fopen(config.filename.c_str(), "wb");
#endif
  if(!m_file)
  {
    LOGE("Video encoder: cannot write %s\n", nvutils::utf8FromPath(config.filename).c_str());
    deinit();
    return false;
  }

  m_timelineValue   = 0;
  m_framesSubmitted = 0;
  m_framesWritten   = 0;
  m_bytesWritten    = 0;
  m_idrPicId        = 0;
  m_rateControlSet  = false;
  return true;
}

bool VideoEncoder::createSession()
{
  VkVideoSessionCreateInfoKHR info{VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR};
  info.queueFamilyIndex           = m_encodeQueue.familyIndex;
  info.pVideoProfile              = &m_profile;
  info.pictureFormat              = NV12_FORMAT;
  info.maxCodedExtent             = m_codedExtent;
  info.referencePictureFormat     = NV12_FORMAT;
  info.maxDpbSlots                = DPB_SLOTS;
  info.maxActiveReferencePictures = 1;
  info.pStdHeaderVersion          = &m_caps.stdHeaderVersion;
  VkResult result                 = vkCreateVideoSessionKHR(m_device, &info, nullptr, &m_session);
  if(result != VK_SUCCESS)
  {
    LOGE("Video encoder: cannot create the video session (%d)\n", result);
    m_session = nullptr;
    return false;
  }

  StdVideoH264SequenceParameterSetVui vui{};
  vui.flags.video_signal_type_present_flag = 1;
  vui.flags.color_description_present_flag = 1;
  vui.flags.timing_info_present_flag       = 1;
  vui.flags.fixed_frame_rate_flag          = 1;
  vui.aspect_ratio_idc                     = STD_VIDEO_H264_ASPECT_RATIO_IDC_UNSPECIFIED;
  vui.video_format                         = 5;  // unspecified
  vui.colour_primaries                     = 1;  // bt709
  vui.transfer_characteristics             = 1;
  vui.matrix_coefficients                  = 1;
  // a frame is two fields of ticks
  vui.num_units_in_tick = 1;
  vui.time_scale        = 2 * m_config.fps;

  StdVideoH264SequenceParameterSet sps{};
  sps.flags.direct_8x8_inference_flag   = 1;
  sps.flags.frame_mbs_only_flag         = 1;
  sps.flags.vui_parameters_present_flag = 1;
  sps.profile_idc                       = STD_VIDEO_H264_PROFILE_IDC_HIGH;
  sps.level_idc                         = m_h264Caps.maxLevelIdc;
  sps.chroma_format_idc                 = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
  sps.log2_max_frame_num_minus4         = 4;
  sps.pic_order_cnt_type                = STD_VIDEO_H264_POC_TYPE_2;  // output order is decode order
  sps.max_num_ref_frames                = 1;
  sps.pic_width_in_mbs_minus1           = m_codedExtent.width / 16 - 1;
  sps.pic_height_in_map_units_minus1    = m_codedExtent.height / 16 - 1;
  sps.pSequenceParameterSetVui          = &vui;
  if(m_codedExtent.width != m_config.size.width || m_codedExtent.height != m_config.size.height)
  {
    // in chroma samples
    sps.flags.frame_cropping_flag = 1;
    sps.frame_crop_right_offset   = (m_codedExtent.width - m_config.size.width) / 2;
    sps.frame_crop_bottom_offset  = (m_codedExtent.height - m_config.size.height) / 2;
  }

  StdVideoH264PictureParameterSet pps{};
  pps.flags.transform_8x8_mode_flag =
      (m_h264Caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_KHR) != 0;
  pps.flags.entropy_coding_mode_flag =
      (m_h264Caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR) != 0;
  pps.weighted_bipred_idc = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT;

  VkVideoEncodeH264SessionParametersAddInfoKHR addInfo{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR};
  addInfo.stdSPSCount = 1;
  addInfo.pStdSPSs    = &sps;
  addInfo.stdPPSCount = 1;
  addInfo.pStdPPSs    = &pps;
  VkVideoEncodeH264SessionParametersCreateInfoKHR h264Info{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR};
  h264Info.maxStdSPSCount     = 1;
  h264Info.maxStdPPSCount     = 1;
  h264Info.pParametersAddInfo = &addInfo;
  VkVideoSessionParametersCreateInfoKHR paramsInfo{VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR};
  paramsInfo.pNext        = &h264Info;
  paramsInfo.videoSession = m_session;
  result                  = vkCreateVideoSessionParametersKHR(m_device, &paramsInfo, nullptr, &m_params);
  if(result != VK_SUCCESS)
  {
    LOGE("Video encoder: cannot create the session parameters (%d)\n", result);
    m_params = nullptr;
    return false;
  }
  return true;
}

bool VideoEncoder::writeHeaders()
{
  // the implementation may have adjusted the parameters, it writes what it encodes with
  VkVideoEncodeH264SessionParametersGetInfoKHR h264Get{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR};
  h264Get.writeStdSPS = VK_TRUE;
  h264Get.writeStdPPS = VK_TRUE;
  VkVideoEncodeSessionParametersGetInfoKHR getInfo{VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR};
  getInfo.pNext                  = &h264Get;
  getInfo.videoSessionParameters = m_params;

  size_t size = 0;
  if(vkGetEncodedVideoSessionParametersKHR(m_device, &getInfo, nullptr, &size, nullptr) != VK_SUCCESS || size == 0)
  {
    LOGE("Video encoder: cannot get the encoded SPS and PPS\n");
    return false;
  }
  m_headers.resize(size);
  if(vkGetEncodedVideoSessionParametersKHR(m_device, &getInfo, nullptr, &size, m_headers.data()) != VK_SUCCESS)
  {
    LOGE("Video encoder: cannot get the encoded SPS and PPS\n");
    return false;
  }
  m_headers.resize(size);
  return true;
}

bool VideoEncoder::deinit()
{
  if(!m_device)
    return true;

  bool ok = true;
  // in submit order
  if(m_session && m_file)
  {
    std::vector<uint32_t> order;
    for(uint32_t i = 0; i < uint32_t(m_slots.size()); i++)
    {
      if(m_slots[i].doneValue)
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return m_slots[a].doneValue < m_slots[b].doneValue; });
    for(uint32_t i : order)
    {
      ok = collect(i) && ok;
    }
  }

  for(Slot& slot : m_slots)
  {
    vkDestroyImageView(m_device, slot.sourceView, nullptr);
    m_allocator->destroyImage(slot.source);
    m_allocator->destroyBuffer(slot.bitstream);
  }
  m_slots.clear();
  m_slotIsIdr.clear();

  vkDestroyImageView(m_device, m_dpbView, nullptr);
  m_dpbView = nullptr;
  if(m_dpb.image)
    m_allocator->destroyImage(m_dpb);

  vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
  vkDestroyQueryPool(m_device, m_queryPool, nullptr);
  vkDestroySemaphore(m_device, m_timeline, nullptr);
  if(m_params)
    vkDestroyVideoSessionParametersKHR(m_device, m_params, nullptr);
  if(m_session)
    vkDestroyVideoSessionKHR(m_device, m_session, nullptr);
  for(VkDeviceMemory memory : m_sessionMemory)
  {
    vkFreeMemory(m_device, memory, nullptr);
  }
  m_sessionMemory.clear();
  m_cmdPool   = nullptr;
  m_queryPool = nullptr;
  m_timeline  = nullptr;
  m_params    = nullptr;
  m_session   = nullptr;

  if(m_file)
  {
    if(fclose(m_file) != 0)
    {
      LOGE("Video encoder: writing %s failed\n", nvutils::utf8FromPath(m_config.filename).c_str());
      ok = false;
    }
    m_file = nullptr;
  }
  m_device = nullptr;
  return ok;
}

void VideoEncoder::cmdUpload(VkCommandBuffer cmd, uint32_t slotIndex, VkBuffer nv12)
{
  Slot& slot = m_slots[slotIndex];
  assert(!slot.doneValue);

  // the previous encode of this picture was waited for on the host
  nvvk::ImageMemoryBarrierParams toTransfer;
  toTransfer.image            = slot.source.image;
  toTransfer.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
  toTransfer.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  toTransfer.srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
  toTransfer.srcAccessMask    = 0;
  toTransfer.dstStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
  toTransfer.dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  nvvk::cmdImageMemoryBarrier(cmd, toTransfer);

  // the padding to whole macroblocks stays undefined, it is cropped
  VkExtent2D        size = m_config.size;
  VkBufferImageCopy regions[2]{};
  regions[0].imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
  regions[0].imageExtent      = {size.width, size.height, 1};
  regions[1].bufferOffset     = VkDeviceSize(size.width) * size.height;
  regions[1].imageSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1};
  regions[1].imageExtent      = {size.width / 2, size.height / 2, 1};
  vkCmdCopyBufferToImage(cmd, nv12, slot.source.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, regions);

  // the frame's semaphore makes the copy visible to the encode queue
  nvvk::ImageMemoryBarrierParams toEncode;
  toEncode.image            = slot.source.image;
  toEncode.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toEncode.newLayout        = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
  toEncode.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  toEncode.srcStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT;
  toEncode.srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  toEncode.dstStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  toEncode.dstAccessMask    = 0;
  nvvk::cmdImageMemoryBarrier(cmd, toEncode);
}

void VideoEncoder::submit(uint32_t slotIndex, const VkSemaphoreSubmitInfo& wait)
{
  Slot& slot = m_slots[slotIndex];
  assert(!slot.doneValue);

  uint32_t frameInGop    = m_framesSubmitted % m_config.idrPeriod;
  bool     idr           = frameInGop == 0;
  uint32_t setupSlot     = m_framesSubmitted % DPB_SLOTS;
  uint32_t refSlot       = (m_framesSubmitted + DPB_SLOTS - 1) % DPB_SLOTS;
  uint32_t frameNum      = frameInGop % MAX_FRAME_NUM;
  int32_t  poc           = int32_t(frameInGop) * 2;
  m_slotIsIdr[slotIndex] = idr;

  VkCommandBuffer cmd = slot.cmd;
  NVVK_CHECK(vkResetCommandBuffer(cmd, 0));
  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

  // the previous encode wrote the reference this one reads
  nvvk::ImageMemoryBarrierParams dpbBarrier;
  dpbBarrier.image     = m_dpb.image;
  dpbBarrier.oldLayout = m_framesSubmitted == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR;
  dpbBarrier.newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR;
  dpbBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, DPB_SLOTS};
  dpbBarrier.srcStageMask     = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
  dpbBarrier.srcAccessMask    = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;
  dpbBarrier.dstStageMask     = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
  dpbBarrier.dstAccessMask    = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;
  nvvk::cmdImageMemoryBarrier(cmd, dpbBarrier);

  vkCmdResetQueryPool(cmd, m_queryPool, slotIndex, 1);

  VkVideoPictureResourceInfoKHR dpbPictures[DPB_SLOTS];
  for(uint32_t i = 0; i < DPB_SLOTS; i++)
  {
    dpbPictures[i]                  = {VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR};
    dpbPictures[i].codedExtent      = m_codedExtent;
    dpbPictures[i].baseArrayLayer   = i;
    dpbPictures[i].imageViewBinding = m_dpbView;
  }

  StdVideoEncodeH264ReferenceInfo setupStd{};
  setupStd.primary_pic_type = idr ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
  setupStd.FrameNum         = frameNum;
  setupStd.PicOrderCnt      = poc;
  VkVideoEncodeH264DpbSlotInfoKHR setupH264{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR};
  setupH264.pStdReferenceInfo = &setupStd;

  StdVideoEncodeH264ReferenceInfo refStd{};
  refStd.primary_pic_type = (frameInGop == 1) ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
  refStd.FrameNum         = m_refFrameNum;
  refStd.PicOrderCnt      = m_refPoc;
  VkVideoEncodeH264DpbSlotInfoKHR refH264{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR};
  refH264.pStdReferenceInfo = &refStd;

  // [0] the picture written, [1] the one referenced
  VkVideoReferenceSlotInfoKHR slots[2] = {{VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR},
                                          {VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR}};
  slots[0].pNext                       = &setupH264;
  slots[0].slotIndex                   = int32_t(setupSlot);
  slots[0].pPictureResource            = &dpbPictures[setupSlot];
  slots[1].pNext                       = &refH264;
  slots[1].slotIndex                   = int32_t(refSlot);
  slots[1].pPictureResource            = &dpbPictures[refSlot];

  // the written picture is bound without a slot, it becomes active with this encode
  VkVideoReferenceSlotInfoKHR beginSlots[2] = {slots[0], slots[1]};
  beginSlots[0].slotIndex                   = -1;
  beginSlots[0].pNext                       = nullptr;
  beginSlots[1].pNext                       = nullptr;

  VkVideoEncodeRateControlInfoKHR rateControl{VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR};
  rateControl.rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;

  VkVideoBeginCodingInfoKHR codingInfo{VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR};
  codingInfo.pNext                  = m_rateControlSet ? &rateControl : nullptr;
  codingInfo.videoSession           = m_session;
  codingInfo.videoSessionParameters = m_params;
  codingInfo.referenceSlotCount     = idr ? 1 : 2;
  codingInfo.pReferenceSlots        = beginSlots;
  vkCmdBeginVideoCodingKHR(cmd, &codingInfo);

  if(!m_rateControlSet)
  {
    VkVideoCodingControlInfoKHR control{VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR};
    control.pNext = &rateControl;
    control.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR | VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR;
    vkCmdControlVideoCodingKHR(cmd, &control);
    m_rateControlSet = true;
  }

  StdVideoEncodeH264ReferenceListsInfo refLists{};
  std::fill(std::begin(refLists.RefPicList0), std::end(refLists.RefPicList0), NO_REFERENCE);
  std::fill(std::begin(refLists.RefPicList1), std::end(refLists.RefPicList1), NO_REFERENCE);
  if(!idr)
    refLists.RefPicList0[0] = uint8_t(refSlot);

  StdVideoEncodeH264SliceHeader sliceHeader{};
  sliceHeader.slice_type                    = idr ? STD_VIDEO_H264_SLICE_TYPE_I : STD_VIDEO_H264_SLICE_TYPE_P;
  sliceHeader.cabac_init_idc                = STD_VIDEO_H264_CABAC_INIT_IDC_0;
  sliceHeader.disable_deblocking_filter_idc = STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED;  // deblocking on
  VkVideoEncodeH264NaluSliceInfoKHR slice{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR};
  slice.constantQp      = int32_t(m_config.qp);
  slice.pStdSliceHeader = &sliceHeader;

  StdVideoEncodeH264PictureInfo pictureStd{};
  pictureStd.flags.IdrPicFlag   = idr;
  pictureStd.flags.is_reference = 1;
  pictureStd.idr_pic_id         = uint16_t(m_idrPicId);
  pictureStd.primary_pic_type   = setupStd.primary_pic_type;
  pictureStd.frame_num          = frameNum;
  pictureStd.PicOrderCnt        = poc;
  pictureStd.pRefLists          = &refLists;
  VkVideoEncodeH264PictureInfoKHR pictureH264{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR};
  pictureH264.naluSliceEntryCount = 1;
  pictureH264.pNaluSliceEntries   = &slice;
  pictureH264.pStdPictureInfo     = &pictureStd;

  VkVideoPictureResourceInfoKHR source{VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR};
  source.codedExtent      = m_codedExtent;
  source.imageViewBinding = slot.sourceView;

  VkVideoEncodeInfoKHR encodeInfo{VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR};
  encodeInfo.pNext               = &pictureH264;
  encodeInfo.dstBuffer           = slot.bitstream.buffer;
  encodeInfo.dstBufferRange      = slot.bitstream.bufferSize;
  encodeInfo.srcPictureResource  = source;
  encodeInfo.pSetupReferenceSlot = &slots[0];
  encodeInfo.referenceSlotCount  = idr ? 0 : 1;
  encodeInfo.pReferenceSlots     = &slots[1];

  vkCmdBeginQuery(cmd, m_queryPool, slotIndex, 0);
  vkCmdEncodeVideoKHR(cmd, &encodeInfo);
  vkCmdEndQuery(cmd, m_queryPool, slotIndex);

  VkVideoEndCodingInfoKHR endInfo{VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR};
  vkCmdEndVideoCodingKHR(cmd, &endInfo);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR, VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR, VK_ACCESS_2_HOST_READ_BIT);
  NVVK_CHECK(vkEndCommandBuffer(cmd));

  VkSemaphoreSubmitInfo waitInfo = wait;
  waitInfo.stageMask             = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
  VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal.semaphore = m_timeline;
  signal.value     = ++m_timelineValue;
  signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmdInfo.commandBuffer = cmd;
  VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submitInfo.waitSemaphoreInfoCount   = wait.semaphore ? 1 : 0;
  submitInfo.pWaitSemaphoreInfos      = &waitInfo;
  submitInfo.commandBufferInfoCount   = 1;
  submitInfo.pCommandBufferInfos      = &cmdInfo;
  submitInfo.signalSemaphoreInfoCount = 1;
  submitInfo.pSignalSemaphoreInfos    = &signal;
  NVVK_CHECK(vkQueueSubmit2(m_encodeQueue.queue, 1, &submitInfo, nullptr));

  slot.doneValue = m_timelineValue;
  m_refFrameNum  = frameNum;
  m_refPoc       = poc;
  if(idr)
    m_idrPicId = (m_idrPicId + 1) % 65536;
  m_framesSubmitted++;
}

bool VideoEncoder::collect(uint32_t slotIndex)
{
  Slot& slot = m_slots[slotIndex];
  if(!slot.doneValue)
    return true;

  VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  waitInfo.semaphoreCount = 1;
  waitInfo.pSemaphores    = &m_timeline;
  waitInfo.pValues        = &slot.doneValue;
  NVVK_CHECK(vkWaitSemaphores(m_device, &waitInfo, ~0ull));
  slot.doneValue = 0;

  uint32_t results[QUERY_RESULTS] = {};
  VkResult result = vkGetQueryPoolResults(m_device, m_queryPool, slotIndex, 1, sizeof(results), results, sizeof(results),
                                          VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
  if(result != VK_SUCCESS || int32_t(results[2]) != VK_QUERY_RESULT_STATUS_COMPLETE_KHR)
  {
    LOGE("Video encoder: encoding a frame failed (%d, status %d)\n", result, int32_t(results[2]));
    return false;
  }

  NVVK_CHECK(vmaInvalidateAllocation(*m_allocator, slot.bitstream.allocation, 0, VK_WHOLE_SIZE));

  bool ok = true;
  if(m_slotIsIdr[slotIndex])
  {
    ok = fwrite(m_headers.data(), 1, m_headers.size(), m_file) == m_headers.size();
    m_bytesWritten += m_headers.size();
  }
  ok = ok && fwrite(slot.bitstream.mapping + results[0], 1, results[1], m_file) == results[1];
  m_bytesWritten += results[1];
  m_framesWritten++;
  return ok;
}

}  // namespace lodclusters
