/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nvutils/parameter_registry.hpp>

#include "resources.hpp"
#include "video_encoder.hpp"

// 1: the nv12 conversion reads the captured image in a shader instead of a copy of it. The whole
// window then needs a sampled swapchain, nvapp's ApplicationCreateInfo::swapchainImageUsage.
#define FRAME_CAPTURE_READ_IMAGE 0

namespace lodclusters {

// Writes every rendered frame to numbered image files, or pipes it into an ffmpeg process that
// encodes a video, e.g. of a camera path.
//
// The copy of a frame is recorded into the command buffer of its frame cycle and read back when
// that cycle comes around again, nvapp has waited for its completion by then. The images are
// encoded on writer threads, a single one feeds ffmpeg to keep the frame order. When they fall
// behind, the render thread waits rather than dropping frames, so a fixed-step camera path yields
// one file per path step. For ffmpeg the frame is converted to nv12 on the GPU first, which more
// than halves what goes through the readback and the pipe, and spares ffmpeg the conversion.
// The Vulkan Video format instead copies the nv12 frame into a source picture of the encoder,
// whose encode queue reads it once the frame has completed, only the bitstream is read back.
class FrameCapture
{
public:
  enum Format
  {
    FORMAT_PNG,
    FORMAT_JPG,
    FORMAT_FFMPEG,
    FORMAT_VULKAN_VIDEO,
  };

  struct Config
  {
    bool                  enabled   = false;
    std::filesystem::path directory = "framecapture";
    int                   format    = FORMAT_PNG;
    int                   writers   = 4;
    std::string           ffmpeg    = "ffmpeg";
    // encoder options, the input as bt709 yuv420p is set up by the capture
    std::string ffmpegArgs = "-c:v hevc_nvenc -preset p5 -cq 20";
    // real-time camera path playback steps this rate while capturing
    int fps = 60;
    // constant quantizer of the Vulkan Video encoder
    int qp = 20;
    // the whole window with its UI instead of the rendered viewport, not headless
    bool window = false;
  };

  Config config;

  ~FrameCapture() { deinit(); }

  void registerParameters(nvutils::ParameterRegistry& registry);

  // without an encode queue the Vulkan Video format is not available
  void init(Resources& res, uint32_t frameCycleCount, const nvvk::QueueInfo* encodeQueue, bool headless);
  // the device must be idle, writes the outstanding frames
  void deinit();

  // Collects the frame this cycle copied last time, then copies `image` (in `layout`, left there)
  // when `config.enabled`. Starts and stops the capture. `format` is an rgba8 or bgra8 format.
  // `frameSignal` is what the frame's submit signals, the encode queue waits for it.
  void cmdFrame(VkCommandBuffer            cmd,
                uint32_t                   cycleIndex,
                VkImage                    image,
                VkImageView                view,
                VkImageLayout              layout,
                VkFormat                   format,
                VkExtent2D                 size,
                const nvvk::SemaphoreInfo& frameSignal);

  // whether `cmdFrame` takes the swapchain image after the UI was drawn, the choice holds for a capture
  bool capturesWindow() const { return (m_active ? m_window : config.window) && !m_headless; }
  // after the frame was submitted, queues the encode of its copy
  void postSubmit();

  // the device must be idle, writes the outstanding frames and ends the capture
  void finish();

  // Every frame before the camera path update, with the size `cmdFrame` will get. In a window a
  // capture waits until the viewport has kept its size for a few frames, the UI layout settles in
  // the first ones, and a camera path holds meanwhile.
  void beginFrame(VkExtent2D size);
  bool isWaiting() const { return m_waiting; }

  bool isActive() const { return m_active; }
  // What time-driven state advances this frame: nothing while waiting, 1/fps while capturing, so a
  // video plays as if rendered at its frame rate. The frame that starts a capture is one of them.
  double getTimeStep(double wallSeconds) const
  {
    return m_waiting ? 0.0 : m_active ? 1.0 / double(m_fps) : config.enabled ? 1.0 / double(config.fps) : wallSeconds;
  }

  void uiSettings();

private:
  struct Slot
  {
    nvvk::Buffer buffer;
    VkExtent2D   size{};
    uint32_t     frameIndex = 0;
    bool         nv12       = false;
    bool         bgra       = false;
    bool         pending    = false;
  };

  struct Job
  {
    std::vector<uint8_t> pixels;
    VkExtent2D           size{};
    uint32_t             frameIndex = 0;
    bool                 nv12       = false;
    bool                 bgra       = false;
  };

  void start();
  bool initNv12();
  void cmdConvertNv12(VkCommandBuffer cmd, VkImage image, VkImageView view, VkImageLayout layout, VkFormat format, VkExtent2D srcSize, VkExtent2D size);
  bool startEncoder(VkExtent2D size);
  void stopWriters();
  void collect(Slot& slot);
  void writerLoop();
  bool writeImage(Job& job);
  bool writeVideoFrame(const Job& job);

  Resources*        m_resources = nullptr;
  std::vector<Slot> m_slots;

  // nv12 conversion, shared by the frames in flight, the queue orders them
  Shader                m_nv12Shader;
  VkPipeline            m_nv12Pipeline  = nullptr;
  VkPipelineLayout      m_nv12Layout    = nullptr;
  VkDescriptorSetLayout m_nv12SetLayout = nullptr;
  nvvk::Buffer          m_rgbaBuffer;
  nvvk::Buffer          m_nv12Buffer;

  bool                  m_active = false;  // writers running, frames may be pending
  Format                m_format = FORMAT_PNG;
  std::filesystem::path m_directory;
  uint32_t              m_framesCopied = 0;
  VkExtent2D            m_firstSize{};
  VkExtent2D            m_lastSize{};  // of the source, to log a change once
  VkExtent2D            m_settleSize{};
  uint32_t              m_settleFrames = 0;
  uint32_t              m_waitedFrames = 0;
  bool                  m_waiting      = false;
  bool                  m_headless     = false;
  bool                  m_window       = false;

  std::vector<std::thread>          m_writers;
  std::mutex                        m_mutex;
  std::condition_variable           m_jobAdded;
  std::condition_variable           m_jobTaken;
  std::deque<Job>                   m_jobs;
  std::vector<std::vector<uint8_t>> m_recycled;
  size_t                            m_maxJobs     = 0;
  bool                              m_stopWriters = false;
  uint32_t                          m_written     = 0;
  // writers read it to log only the first failure
  std::atomic<uint32_t> m_failed = 0;

  // seconds over the capture: render thread waiting for queue space and copying out of the
  // readback buffer, writers encoding and writing
  double m_waitSeconds  = 0;
  double m_copySeconds  = 0;
  double m_writeSeconds = 0;

  // only touched by the single writer of FORMAT_FFMPEG, and after it was joined
  FILE*       m_pipe = nullptr;
  VkExtent2D  m_pipeSize{};
  bool        m_pipeNv12 = false;
  bool        m_pipeBgra = false;
  std::string m_ffmpeg;
  std::string m_ffmpegArgs;
  int         m_fps = 60;  // of the running capture

  // FORMAT_VULKAN_VIDEO, on the render thread
  VideoEncoder          m_encoder;
  nvvk::QueueInfo       m_encodeQueue{};
  bool                  m_hasEncodeQueue = false;
  int32_t               m_submitSlot     = -1;  // copied, its encode not yet submitted
  VkSemaphoreSubmitInfo m_submitWait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};

  char m_directoryEdit[512]  = {};
  char m_ffmpegEdit[512]     = {};
  char m_ffmpegArgsEdit[512] = {};
};

}  // namespace lodclusters
