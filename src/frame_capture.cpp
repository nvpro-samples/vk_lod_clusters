/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#ifndef _WIN32
#include <csignal>
#endif

#include <volk.h>
#include <fmt/format.h>
#include <imgui/imgui.h>
#include <nvgui/property_editor.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>

// nvpro_core2 compiles its copies static, so does this one
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIW_WINDOWS_UTF8
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include <stb/stb_image_write.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "frame_capture.hpp"
#include "../shaders/shaderio_frame_capture.h"

namespace lodclusters {

static constexpr int MAX_WRITERS = 8;
// frames a windowed viewport keeps its size before a capture starts
static constexpr uint32_t SETTLE_FRAMES = 4;
// a viewport that keeps changing does not hold the capture longer
static constexpr uint32_t MAX_WAIT_FRAMES = 30;

void FrameCapture::registerParameters(nvutils::ParameterRegistry& registry)
{
  registry.add({"framecapture", "write every rendered frame to numbered image files or a video in framecapturedir. default false"},
               &config.enabled);
  registry.add({"framecapturedir", "directory of the captured frames, created if missing. default framecapture"},
               &config.directory);
  registry.add({"framecaptureformat", "0 png, 1 jpg, 2 video.mp4 encoded by ffmpeg, 3 video.h264 encoded by Vulkan Video. default 0"},
               &config.format, 0, 3);
  registry.add({"framecapturewriters", "threads encoding png or jpg files. 1..8, default 4"}, &config.writers, 1, MAX_WRITERS);
  registry.add({"framecaptureffmpeg", "ffmpeg executable for framecaptureformat 2. default ffmpeg, from the PATH"},
               &config.ffmpeg);
  registry.add({"framecaptureffmpegargs", "ffmpeg encoder options for framecaptureformat 2. default \"-c:v hevc_nvenc -preset p5 -cq 20\""},
               &config.ffmpegArgs);
  registry.add({"framecapturefps",
                "while capturing, real-time camera path playback advances 1/fps per frame, so the frames play at the "
                "path's speed. 1..240, default 60"},
               &config.fps, 1, 240);
  registry.add({"framecapturewindow", "capture the whole window with its UI instead of the rendered viewport, not headless. default false"},
               &config.window);
  registry.add({"framecaptureqp", "constant quantizer of framecaptureformat 3, lower is better quality. 1..51, default 20"},
               &config.qp, 1, 51);
}

void FrameCapture::init(Resources& res, uint32_t frameCycleCount, const nvvk::QueueInfo* encodeQueue, bool headless)
{
  m_headless       = headless;
  m_resources      = &res;
  m_hasEncodeQueue = encodeQueue != nullptr;
  m_encodeQueue    = encodeQueue ? *encodeQueue : nvvk::QueueInfo{};
  m_slots.resize(frameCycleCount);
}

void FrameCapture::deinit()
{
  if(!m_resources)
    return;

  finish();
  for(Slot& slot : m_slots)
  {
    m_resources->m_allocator.destroyBuffer(slot.buffer);
  }
  m_slots.clear();
  m_resources->m_allocator.destroyBuffer(m_rgbaBuffer);
  m_resources->m_allocator.destroyBuffer(m_nv12Buffer);
  vkDestroyPipeline(m_resources->m_device, m_nv12Pipeline, nullptr);
  vkDestroyPipelineLayout(m_resources->m_device, m_nv12Layout, nullptr);
  vkDestroyDescriptorSetLayout(m_resources->m_device, m_nv12SetLayout, nullptr);
  m_nv12SetLayout = nullptr;
  m_nv12Pipeline  = nullptr;
  m_nv12Layout    = nullptr;
  m_resources     = nullptr;
}

bool FrameCapture::initNv12()
{
  Resources&              res     = *m_resources;
  shaderc::CompileOptions options = res.makeCompilerOptions();
  options.AddMacroDefinition("FRAME_CAPTURE_READ_IMAGE", std::to_string(FRAME_CAPTURE_READ_IMAGE));
  res.compileShader(m_nv12Shader, VK_SHADER_STAGE_COMPUTE_BIT, "frame_capture_nv12.comp.glsl", &options);
  if(!res.verifyShaders(1, &m_nv12Shader))
  {
    return false;
  }

  if(!m_nv12Layout)
  {
#if FRAME_CAPTURE_READ_IMAGE
    // the captured image, pushed per frame
    VkDescriptorSetLayoutBinding    binding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
    setInfo.bindingCount = 1;
    setInfo.pBindings    = &binding;
    NVVK_CHECK(vkCreateDescriptorSetLayout(res.m_device, &setInfo, nullptr, &m_nv12SetLayout));
    NVVK_CHECK(nvvk::createPipelineLayout(res.m_device, &m_nv12Layout, {m_nv12SetLayout},
                                          {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::FrameCaptureNv12Constants)}}));
#else
    NVVK_CHECK(nvvk::createPipelineLayout(res.m_device, &m_nv12Layout, {},
                                          {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::FrameCaptureNv12Constants)}}));
#endif
  }

  VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
  compInfo.stage.pName                 = "main";
  compInfo.layout                      = m_nv12Layout;
  res.createComputePipeline(compInfo, m_nv12Shader, m_nv12Pipeline);
  return m_nv12Pipeline != nullptr;
}

void FrameCapture::start()
{
  std::error_code ec;
  std::filesystem::create_directories(config.directory, ec);
  if(ec)
  {
    LOGE("Frame capture: cannot create %s: %s\n", nvutils::utf8FromPath(config.directory).c_str(), ec.message().c_str());
    config.enabled = false;
    return;
  }

  m_directory  = config.directory;
  m_format     = Format(std::clamp(config.format, 0, 3));
  m_ffmpeg     = config.ffmpeg;
  m_ffmpegArgs = config.ffmpegArgs;
  m_fps        = config.fps;
  m_window     = config.window && !m_headless;
  if(config.window && m_headless)
    LOGW("Frame capture: a headless run has no window, it captures the viewport\n");
  m_framesCopied = 0;
  m_firstSize    = {};
  m_written      = 0;
  m_failed       = 0;
  m_stopWriters  = false;
  m_waitSeconds  = 0;
  m_copySeconds  = 0;
  m_writeSeconds = 0;

  m_submitSlot = -1;

  if(m_format == FORMAT_VULKAN_VIDEO && !m_hasEncodeQueue)
  {
    LOGW("Frame capture: the device has no queue for H.264 encoding, nothing is captured\n");
    config.enabled = false;
    return;
  }

  bool video = m_format == FORMAT_FFMPEG || m_format == FORMAT_VULKAN_VIDEO;
  if(video && !m_nv12Pipeline && !initNv12())
  {
    if(m_format == FORMAT_VULKAN_VIDEO)
    {
      LOGE("Frame capture: the nv12 conversion did not compile\n");
      config.enabled = false;
      return;
    }
    LOGW("Frame capture: the nv12 conversion did not compile, ffmpeg converts the frames instead\n");
  }

  // the encoder needs no writer, one feeds ffmpeg in order
  int numWriters = m_format == FORMAT_VULKAN_VIDEO ? 0 :
                   m_format == FORMAT_FFMPEG       ? 1 :
                                                     std::clamp(config.writers, 1, MAX_WRITERS);
  // the writers wait on the render thread, which waits on them beyond this
  m_maxJobs = size_t(std::max(numWriters, 1)) * 2;

  // stb's per-row filter search costs a third more encode time than a fixed filter, for files no smaller
  stbi_write_png_compression_level = 2;
  stbi_write_force_png_filter      = 1;

#ifndef _WIN32
  // an ffmpeg that failed to start or exited early must fail the write, not end the process
  signal(SIGPIPE, SIG_IGN);
#endif

  for(int i = 0; i < numWriters; i++)
  {
    m_writers.emplace_back(&FrameCapture::writerLoop, this);
  }
  m_active = true;

  LOGI("Frame capture: writing to %s\n", nvutils::utf8FromPath(m_directory).c_str());
}

void FrameCapture::stopWriters()
{
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stopWriters = true;
  }
  m_jobAdded.notify_all();
  for(std::thread& writer : m_writers)
  {
    writer.join();
  }
  m_writers.clear();
  m_recycled.clear();
  m_active = false;

  bool ffmpegFailed = false;
  if(m_pipe)
  {
#ifdef _WIN32
    int exitCode = _pclose(m_pipe);
#else
    int exitCode = pclose(m_pipe);
#endif
    m_pipe       = nullptr;
    ffmpegFailed = exitCode != 0;
    if(ffmpegFailed)
    {
      LOGE("Frame capture: ffmpeg exited with %d, see its output above\n", exitCode);
    }
  }

  bool videoFailed = ffmpegFailed;
  if(m_encoder.isValid())
  {
    LOGI("Frame capture: %.1f MB of H.264\n", double(m_encoder.getBytesWritten()) / (1024.0 * 1024.0));
    videoFailed = !m_encoder.deinit() || videoFailed;
  }

  if(m_failed)
    LOGE("Frame capture: %u of %u frames could not be written\n", m_failed.load(), m_framesCopied);
  if(!videoFailed)
    LOGI("Frame capture: %u frames written to %s\n", m_written, nvutils::utf8FromPath(m_directory).c_str());
  if(m_framesCopied)
  {
    double toMs = 1000.0 / double(m_framesCopied);
    LOGI("Frame capture: per frame %.2f ms waiting for the writers, %.2f ms copying, %.2f ms writing\n",
         m_waitSeconds * toMs, m_copySeconds * toMs, m_writeSeconds * toMs);
  }
}

void FrameCapture::finish()
{
  postSubmit();
  // in frame order, the video writer takes them as they come
  std::vector<Slot*> pending;
  for(Slot& slot : m_slots)
  {
    if(slot.pending)
      pending.push_back(&slot);
  }
  std::sort(pending.begin(), pending.end(), [](const Slot* a, const Slot* b) { return a->frameIndex < b->frameIndex; });
  for(Slot* slot : pending)
  {
    collect(*slot);
  }
  if(m_active)
  {
    stopWriters();
  }
  config.enabled = false;
}

void FrameCapture::collect(Slot& slot)
{
  if(!slot.pending)
    return;
  slot.pending = false;

  if(m_format == FORMAT_VULKAN_VIDEO)
  {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    bool                                  ok    = m_encoder.collect(uint32_t(&slot - m_slots.data()));
    m_writeSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    if(ok)
      m_written++;
    else
      m_failed++;
    return;
  }

  size_t pixelCount = size_t(slot.size.width) * slot.size.height;
  size_t bytes      = slot.nv12 ? pixelCount * 3 / 2 : pixelCount * 4;

  std::vector<uint8_t> pixels;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_jobTaken.wait(lock, [&] { return m_jobs.size() < m_maxJobs; });
    if(!m_recycled.empty())
    {
      pixels = std::move(m_recycled.back());
      m_recycled.pop_back();
    }
  }
  std::chrono::steady_clock::time_point waited = std::chrono::steady_clock::now();

  pixels.resize(bytes);
  memcpy(pixels.data(), slot.buffer.mapping, bytes);

  m_waitSeconds += std::chrono::duration<double>(waited - begin).count();
  m_copySeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - waited).count();

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_jobs.push_back({std::move(pixels), slot.size, slot.frameIndex, slot.nv12, slot.bgra});
  }
  m_jobAdded.notify_one();
}

void FrameCapture::postSubmit()
{
  if(m_submitSlot < 0)
    return;
  m_encoder.submit(uint32_t(m_submitSlot), m_submitWait);
  m_submitSlot = -1;
}

bool FrameCapture::startEncoder(VkExtent2D size)
{
  VideoEncoder::Config encoderConfig;
  encoderConfig.size      = size;
  encoderConfig.slotCount = uint32_t(m_slots.size());
  encoderConfig.fps       = uint32_t(m_fps);
  encoderConfig.qp        = uint32_t(config.qp);
  encoderConfig.idrPeriod = uint32_t(m_fps) * 2;
  encoderConfig.filename  = m_directory / "video.h264";
  return m_encoder.init(m_resources->m_device, m_resources->m_physicalDevice, m_resources->m_allocator,
                        m_resources->m_queue, m_encodeQueue, encoderConfig);
}

void FrameCapture::beginFrame(VkExtent2D size)
{
  if(size.width != m_settleSize.width || size.height != m_settleSize.height)
  {
    m_settleSize   = size;
    m_settleFrames = 0;
  }
  m_settleFrames = std::min(m_settleFrames + 1, SETTLE_FRAMES);
  m_waiting = config.enabled && !m_active && !m_headless && m_settleFrames < SETTLE_FRAMES && m_waitedFrames < MAX_WAIT_FRAMES;
  m_waitedFrames = m_waiting ? m_waitedFrames + 1 : 0;
}

void FrameCapture::cmdFrame(VkCommandBuffer            cmd,
                            uint32_t                   cycleIndex,
                            VkImage                    image,
                            VkImageView                view,
                            VkImageLayout              layout,
                            VkFormat                   format,
                            VkExtent2D                 size,
                            const nvvk::SemaphoreInfo& frameSignal)
{
  assert(cycleIndex < m_slots.size());
  Slot& slot = m_slots[cycleIndex];

  // in case the previous frame was not followed by postSubmit
  postSubmit();
  collect(slot);

  if(!config.enabled)
  {
    // a stop lets the frames in flight arrive first
    if(m_active && std::none_of(m_slots.begin(), m_slots.end(), [](const Slot& s) { return s.pending; }))
    {
      stopWriters();
    }
    return;
  }

  bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
  if(!bgra && format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_R8G8B8A8_SRGB)
  {
    LOGE("Frame capture: the image format %d is not rgba8 or bgra8\n", int(format));
    config.enabled = false;
    return;
  }

  if(!m_active)
  {
    if(m_waiting)
      return;
    start();
    if(!m_active)
      return;
  }

  bool video = m_format == FORMAT_FFMPEG || m_format == FORMAT_VULKAN_VIDEO;
  // nv12 takes whole words of luma and an even height, a viewport can have any size
  if(video)
  {
    size = {size.width & ~3u, size.height & ~1u};
  }

  if(m_framesCopied == 0)
  {
    m_firstSize = size;
    m_lastSize  = size;
  }
  else if(size.width != m_lastSize.width || size.height != m_lastSize.height)
  {
    LOGW("Frame capture: frame %u is %ux%u, %s %ux%u\n", m_framesCopied, size.width, size.height,
         video && m_nv12Pipeline ? "scaled to fit the video's" : "the capture started at", m_firstSize.width,
         m_firstSize.height);
    m_lastSize = size;
  }

  // a video keeps its size, the conversion scales a resized frame to fit
  bool       nv12    = video && m_nv12Pipeline;
  VkExtent2D srcSize = size;
  if(nv12)
    size = m_firstSize;

  if(m_format == FORMAT_VULKAN_VIDEO && !m_encoder.isValid())
  {
    if(!startEncoder(size))
    {
      config.enabled = false;
      stopWriters();
      return;
    }
  }

  VkDeviceSize pixelCount = VkDeviceSize(size.width) * size.height;
  VkDeviceSize bytes      = nv12 ? pixelCount * 3 / 2 : pixelCount * 4;
  if(m_format != FORMAT_VULKAN_VIDEO && slot.buffer.bufferSize < bytes)
  {
    // this cycle's previous copy has completed
    m_resources->m_allocator.destroyBuffer(slot.buffer);
    NVVK_CHECK(m_resources->m_allocator.createBuffer(slot.buffer, bytes, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                                     VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
  }

  slot.size       = size;
  slot.frameIndex = m_framesCopied++;
  slot.nv12       = nv12;
  slot.bgra       = bgra;
  slot.pending    = true;

  if(nv12)
  {
    cmdConvertNv12(cmd, image, view, layout, format, srcSize, size);
    if(m_format == FORMAT_VULKAN_VIDEO)
    {
      m_encoder.cmdUpload(cmd, cycleIndex, m_nv12Buffer.buffer);
      m_submitSlot           = int32_t(cycleIndex);
      m_submitWait           = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      m_submitWait.semaphore = frameSignal.semaphore;
      m_submitWait.value     = frameSignal.value;
      return;
    }

    VkBufferCopy copy{0, 0, pixelCount * 3 / 2};
    vkCmdCopyBuffer(cmd, m_nv12Buffer.buffer, slot.buffer.buffer, 1, &copy);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_HOST_READ_BIT);
    return;
  }

  nvvk::ImageMemoryBarrierParams toTransfer;
  toTransfer.image         = image;
  toTransfer.oldLayout     = layout;
  toTransfer.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  toTransfer.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  toTransfer.srcAccessMask = 0;
  nvvk::cmdImageMemoryBarrier(cmd, toTransfer);

  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent      = {size.width, size.height, 1};
  vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.buffer.buffer, 1, &region);

  nvvk::ImageMemoryBarrierParams toShader;
  toShader.image     = image;
  toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  toShader.newLayout = layout;
  nvvk::cmdImageMemoryBarrier(cmd, toShader);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_HOST_READ_BIT);
}

void FrameCapture::cmdConvertNv12(VkCommandBuffer cmd, VkImage image, VkImageView view, VkImageLayout layout, VkFormat format, VkExtent2D srcSize, VkExtent2D size)
{
  bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
#if FRAME_CAPTURE_READ_IMAGE
  // an srgb view would decode the values the conversion takes as they are
  bool readImage = view && (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_B8G8R8A8_UNORM);
#else
  bool readImage = false;
#endif

  VkDeviceSize srcPixelCount = readImage ? 0 : VkDeviceSize(srcSize.width) * srcSize.height;
  VkDeviceSize pixelCount    = VkDeviceSize(size.width) * size.height;
  if(m_rgbaBuffer.bufferSize < srcPixelCount * 4 || m_nv12Buffer.bufferSize < pixelCount * 3 / 2)
  {
    // frames of the other cycles may still use them, a size change is rare enough to wait
    NVVK_CHECK(vkDeviceWaitIdle(m_resources->m_device));
    m_resources->m_allocator.destroyBuffer(m_nv12Buffer);
    NVVK_CHECK(m_resources->m_allocator.createBuffer(m_nv12Buffer, pixelCount * 3 / 2,
                                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
                                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
    if(m_rgbaBuffer.bufferSize < srcPixelCount * 4)
    {
      m_resources->m_allocator.destroyBuffer(m_rgbaBuffer);
      NVVK_CHECK(m_resources->m_allocator.createBuffer(m_rgbaBuffer, srcPixelCount * 4, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
    }
  }

  if(readImage)
  {
#if FRAME_CAPTURE_READ_IMAGE
    // the previous frame may still read the shared buffer
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    if(layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
      nvvk::ImageMemoryBarrierParams toShader;
      toShader.image         = image;
      toShader.oldLayout     = layout;
      toShader.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      toShader.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      toShader.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
      toShader.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      toShader.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
      nvvk::cmdImageMemoryBarrier(cmd, toShader);
    }

    VkDescriptorImageInfo imageInfo{nullptr, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet  write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo      = &imageInfo;
    vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nv12Layout, 0, 1, &write);
#endif
  }
  else
  {
    // the previous frame may still read the shared buffers
    nvvk::ImageMemoryBarrierParams toTransfer;
    toTransfer.image         = image;
    toTransfer.oldLayout     = layout;
    toTransfer.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransfer.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    toTransfer.srcAccessMask = 0;
    nvvk::cmdImageMemoryBarrier(cmd, toTransfer);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 0);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {srcSize.width, srcSize.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_rgbaBuffer.buffer, 1, &region);

    nvvk::ImageMemoryBarrierParams toShader;
    toShader.image     = image;
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toShader.newLayout = layout;
    nvvk::cmdImageMemoryBarrier(cmd, toShader);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
  }

  shaderio::FrameCaptureNv12Constants constants;
  constants.srcAddress = m_rgbaBuffer.address;
  constants.dstAddress = m_nv12Buffer.address;
  constants.width      = size.width;
  constants.height     = size.height;
  constants.srcWidth   = srcSize.width;
  constants.srcHeight  = srcSize.height;
  float scale          = std::min(float(size.width) / float(srcSize.width), float(size.height) / float(srcSize.height));
  constants.invScale   = 1.0f / scale;
  constants.offset     = {(float(size.width) - float(srcSize.width) * scale) * 0.5f,
                          (float(size.height) - float(srcSize.height) * scale) * 0.5f};
  constants.readImage  = readImage ? 1 : 0;
  // a view of a bgra image returns rgb already
  constants.swapRedBlue = bgra && !readImage ? 1 : 0;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nv12Pipeline);
  vkCmdPushConstants(cmd, m_nv12Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
  vkCmdDispatch(cmd, (size.width / 4 + FRAME_CAPTURE_NV12_WORKGROUP_X - 1) / FRAME_CAPTURE_NV12_WORKGROUP_X,
                (size.height / 2 + FRAME_CAPTURE_NV12_WORKGROUP_Y - 1) / FRAME_CAPTURE_NV12_WORKGROUP_Y, 1);

  if(readImage && layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    nvvk::ImageMemoryBarrierParams back;
    back.image         = image;
    back.oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    back.newLayout     = layout;
    back.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    back.srcAccessMask = 0;
    nvvk::cmdImageMemoryBarrier(cmd, back);
  }

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
}

void FrameCapture::writerLoop()
{
  for(;;)
  {
    Job job;
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_jobAdded.wait(lock, [&] { return !m_jobs.empty() || m_stopWriters; });
      if(m_jobs.empty())
        return;
      job = std::move(m_jobs.front());
      m_jobs.pop_front();
    }
    m_jobTaken.notify_one();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    bool                                  ok    = m_format == FORMAT_FFMPEG ? writeVideoFrame(job) : writeImage(job);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

    std::lock_guard<std::mutex> lock(m_mutex);
    m_writeSeconds += seconds;
    if(ok)
      m_written++;
    else
      m_failed++;
    m_recycled.push_back(std::move(job.pixels));
  }
}

bool FrameCapture::writeImage(Job& job)
{
  // RGBA or BGRA to RGB in place, the alpha channel is not meaningful
  uint8_t* pixels    = job.pixels.data();
  size_t   numPixels = size_t(job.size.width) * job.size.height;
  size_t   red       = job.bgra ? 2 : 0;
  for(size_t i = 0; i < numPixels; i++)
  {
    uint8_t r         = pixels[i * 4 + red];
    uint8_t g         = pixels[i * 4 + 1];
    uint8_t b         = pixels[i * 4 + 2 - red];
    pixels[i * 3 + 0] = r;
    pixels[i * 3 + 1] = g;
    pixels[i * 3 + 2] = b;
  }

  int         width  = int(job.size.width);
  int         height = int(job.size.height);
  std::string filename =
      nvutils::utf8FromPath(m_directory / fmt::format("frame_{:08}.{}", job.frameIndex, m_format == FORMAT_PNG ? "png" : "jpg"));
  int ok = m_format == FORMAT_PNG ? stbi_write_png(filename.c_str(), width, height, 3, pixels, width * 3) :
                                    stbi_write_jpg(filename.c_str(), width, height, 3, pixels, 95);
  if(!ok && m_failed == 0)
  {
    LOGE("Frame capture: could not write %s\n", filename.c_str());
  }
  return ok != 0;
}

bool FrameCapture::writeVideoFrame(const Job& job)
{
  if(!m_pipe && m_failed == 0)
  {
    // the size and format are known with the first frame
    m_pipeSize           = job.size;
    m_pipeNv12           = job.nv12;
    m_pipeBgra           = job.bgra;
    std::string filename = nvutils::utf8FromPath(m_directory / "video.mp4");
    // Players assume bt709 for HD content, ffmpeg's rgb to yuv conversion defaults to bt601.
    // The tags go through setparams, ffmpeg drops the output options with some encoders.
    const char* input   = job.nv12 ? "nv12" : job.bgra ? "bgra" : "rgba";
    const char* filter  = job.nv12 ? "" : "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,";
    std::string command = fmt::format(
        "\"{}\" -hide_banner -loglevel warning -y -f rawvideo -pix_fmt {} -s {}x{} -framerate {} -i - "
        "-vf {}setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv {} \"{}\"",
        m_ffmpeg, input, job.size.width, job.size.height, m_fps, filter, m_ffmpegArgs, filename);
    LOGI("Frame capture: %s\n", command.c_str());
#ifdef _WIN32
    // cmd.exe strips the outer quotes of a command that starts with one
    std::wstring wcommand = nvutils::pathFromUtf8("\"" + command + "\"").wstring();
    m_pipe                = _wpopen(wcommand.c_str(), L"wb");
#else
    m_pipe = popen(command.c_str(), "w");
#endif
    if(!m_pipe)
    {
      LOGE("Frame capture: could not start ffmpeg\n");
      return false;
    }
  }
  if(!m_pipe)
    return false;

  if(job.size.width != m_pipeSize.width || job.size.height != m_pipeSize.height || job.nv12 != m_pipeNv12
     || (!job.nv12 && job.bgra != m_pipeBgra))
  {
    if(m_failed == 0)
      LOGE("Frame capture: frame %u is %ux%u, the video is %ux%u, skipped\n", job.frameIndex, job.size.width,
           job.size.height, m_pipeSize.width, m_pipeSize.height);
    return false;
  }

  size_t pixelCount = size_t(job.size.width) * job.size.height;
  size_t bytes      = job.nv12 ? pixelCount * 3 / 2 : pixelCount * 4;
  if(fwrite(job.pixels.data(), 1, bytes, m_pipe) != bytes)
  {
    if(m_failed == 0)
    {
      LOGE("Frame capture: writing to ffmpeg failed, see its output above\n");
    }
    return false;
  }
  return true;
}

void FrameCapture::uiSettings()
{
  namespace PE = nvgui::PropertyEditor;

  PE::begin("framecapture", ImGuiTableFlags_Resizable);
  PE::Checkbox("Capture frames", &config.enabled,
               "Write every rendered frame to numbered image files or a video, e.g. while a camera path plays. "
               "Rendering waits for the writers.");

  ImGui::BeginDisabled(m_active);
  if(!ImGui::IsAnyItemActive() || m_directoryEdit[0] == 0)
  {
    std::string directory = nvutils::utf8FromPath(config.directory);
    snprintf(m_directoryEdit, sizeof(m_directoryEdit), "%s", directory.c_str());
  }
  if(PE::InputText("Directory", m_directoryEdit, sizeof(m_directoryEdit), 0, "Created if missing, files are named frame_<index>"))
  {
    config.directory = nvutils::pathFromUtf8(m_directoryEdit);
  }
  PE::Checkbox("Whole window", &config.window, "The window with its UI instead of the rendered viewport");
  PE::Combo("Format", &config.format, "png\0jpg\0video.mp4 (ffmpeg)\0video.h264 (Vulkan Video)\0\0");
  if(config.format == FORMAT_FFMPEG)
  {
    if(!ImGui::IsAnyItemActive() || m_ffmpegEdit[0] == 0)
    {
      snprintf(m_ffmpegEdit, sizeof(m_ffmpegEdit), "%s", config.ffmpeg.c_str());
      snprintf(m_ffmpegArgsEdit, sizeof(m_ffmpegArgsEdit), "%s", config.ffmpegArgs.c_str());
    }
    if(PE::InputText("ffmpeg", m_ffmpegEdit, sizeof(m_ffmpegEdit), 0, "Executable, from the PATH unless a path is given"))
      config.ffmpeg = m_ffmpegEdit;
    if(PE::InputText("ffmpeg options", m_ffmpegArgsEdit, sizeof(m_ffmpegArgsEdit), 0,
                     "Encoder options, the input as bt709 yuv420p is set up by the capture"))
      config.ffmpegArgs = m_ffmpegArgsEdit;
  }
  else if(config.format == FORMAT_VULKAN_VIDEO)
  {
    PE::SliderInt("QP", &config.qp, 1, 51, "%d", 0, "Constant quantizer, lower is better quality");
  }
  else
  {
    PE::SliderInt("Writer threads", &config.writers, 1, MAX_WRITERS, "%d", 0, "Threads encoding the image files");
  }
  PE::SliderInt("Real-time fps", &config.fps, 1, 240, "%d", 0,
                "Frame rate of the video. While capturing, real-time camera path playback advances 1/fps per frame");
  ImGui::EndDisabled();
  PE::end();

  if(m_active)
  {
    uint32_t written;
    size_t   queued;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      written = m_written;
      queued  = m_jobs.size();
    }
    ImGui::Text("%u frames, %u written, %zu queued", m_framesCopied, written, queued);
  }
}

}  // namespace lodclusters
