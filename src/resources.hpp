/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <glm/glm.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/physical_device.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvkglsl/glsl.hpp>
#include <vk_radix_sort.h>

#if VK_HEADER_VERSION < 309
#error Update Vulkan SDK >= 1.4.309.0
#endif

#if USE_DLSS
#include "dlss_denoiser.hpp"
#include "dlss_upscaler.hpp"
#endif

#include <nvvk/async_uploader.hpp>
#include "hbao_pass.hpp"
#include "nvhiz_vk.hpp"
#include "../shaders/shaderio.h"

namespace lodclusters {

// a compiled shader and the file it came from, the file name is used when dumping
// pipeline internals. Implicitly converts, so it can be passed to the nvvkglsl getters.
struct Shader
{
  shaderc::SpvCompilationResult compiled;
  std::filesystem::path         fileName;

  operator shaderc::SpvCompilationResult&() { return compiled; }
  operator const shaderc::SpvCompilationResult&() const { return compiled; }
};

struct FrameConfig
{
  VkExtent2D windowSize;

  bool  showInstanceBboxes = false;
  bool  showClusterBboxes  = false;
  bool  freezeCulling      = false;
  bool  freezeLoD          = false;
  bool  freezeStreaming    = false;
  bool  hbaoActive         = true;
  bool  adaptiveError      = false;
  float lodPixelError      = 1.0f;

  // increase error by this for instances not having primary visibility in ray tracing
  float culledErrorScale = 2.0f;
  // if less pixels than this, use sw raster
  float swRasterThreshold = 8.0f;

  // how many frames until we schedule a group for unloading
  uint32_t streamingAgeThreshold    = 16;
  float    streamingUnloadThreshold = 0.0f;

  // all-levels-rolling propagation (STREAM_PROPAGATION_ALL_LEVELS_ROLLING): number of coarsest
  // lod levels swept in full every frame (cheap - few groups - giving coarse data single-frame
  // stream-in reach); the remaining finer levels advance one per frame. >= 1; >= lod levels
  // reproduces the full sweep.
  uint32_t streamRollingCoarseLevels = 4;

  // scales the acceptable error derived from hit ray-cone widths when
  // hit-based lod clamping is active (RendererConfig::useHitStreamingLod)
  float hitLodWidthFactor = 1.0f;
  // hit-based lod clamping: groups also open while the camera metric at this multiple of
  // the error wants them, regardless of ray hits. 0 disables
  float hitLodCameraScale = 0.0f;

  // how much threads to use in the persistent kernels
  uint32_t traversalPersistentThreads = 2048;

  uint32_t sharingTolerantLevels = 7;
  uint32_t sharingEnabledLevels  = 8;
  bool     sharingPushCulled     = true;

  uint32_t cachingEnabledLevels = 8;
  uint32_t cachingAgeThreshold  = 16;
  // extra lod levels an instance may give up to reach the cached blas, 0 is exact
  uint32_t cachingPushLevels = 0;
  // hysteresis: extra levels for an instance already on the cached blas, keeps partitions quiet
  uint32_t cachingStickyLevels = 0;
  // hybrid path: a visible instance keeps a cached blas within N levels of its cut, -1 off
  int32_t hybridCachedTolerance = -1;
  // early-out: a stable partition is reprocessed at least this often. Bounds how stale a
  // skipped partition's lod decision may get, and is also the window over which the
  // cached blas retention value is minimised. 0 uses the default.
  uint32_t partitionRevalidateFrames = 64;
  // early-out: partitions crossing or entering the frustum are processed every frame
  bool partitionFrustumRevalidate = true;
  // early-out: the frustum revalidation stops at this fraction of the scene size
  float partitionFrustumDistance = 0.25f;
  // early-out: step a pinned cached blas down instead of holding its groups forever
  bool blasCacheRetire = true;

  // rasterization: how many tail lod levels may skip the lod traversal and
  // how many lod levels an instance's lod range may span (0 disables the test)
  uint32_t discreteEnabledLevels = 8;
  uint32_t discreteLodRange      = 16;

  HbaoPass::Settings hbaoSettings;

  uint32_t visualize = VISUALIZE_LOD;

  shaderio::FrameConstants frameConstants;
  shaderio::FrameConstants frameConstantsLast;
  float                    traversalFov;
  float                    traversalViewHeight;
  glm::mat4                traversalViewMatrix;
  glm::mat4                cullViewProjMatrix;
  glm::mat4                cullViewProjMatrixLast;
};

//////////////////////////////////////////////////////////////////////////

inline void cmdCopyBuffer(VkCommandBuffer cmd, const nvvk::Buffer& src, const nvvk::Buffer& dst)
{
  VkBufferCopy cpy = {0, 0, src.bufferSize};
  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}

std::string formatMemorySize(size_t sizeInBytes);

inline size_t logMemoryUsage(size_t size, const char* memtype, const char* what)
{
  LOGI("%s memory: %s - %s\n", memtype, formatMemorySize(size).c_str(), what);
  return size;
}

//////////////////////////////////////////////////////////////////////////

struct BufferRanges
{
  VkDeviceSize tempOffset = 0;

  VkDeviceSize beginOffset = 0;
  VkDeviceSize splitOffset = 0;

  VkDeviceSize append(VkDeviceSize size, VkDeviceSize alignment)
  {
    tempOffset = nvutils::align_up(tempOffset, alignment);

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

  VkDeviceSize getSize(VkDeviceSize alignment = 4) { return nvutils::align_up(tempOffset, alignment); }
};

//////////////////////////////////////////////////////////////////////////

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

  nvvk::SemaphoreState getCurrentState() const
  {
    return nvvk::SemaphoreState::makeFixed(m_timelineSemaphore, m_timelineValue);
  }

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
  static constexpr VkPipelineStageFlags2 ALL_SHADER_STAGES =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

#if USE_DLSS
  enum class DlssMode
  {
    eNone,
    eRayReconstruction,
    eSuperResolution,
  };
#endif

  struct FrameBuffer
  {
    static constexpr int MAX_SUPERSAMPLE = 4;

    VkExtent2D renderSize{};
    VkExtent2D targetSize{};
    VkExtent2D windowSize{};
    VkExtent2D targetSizeLast{};
    // window, or the target of a fixed resolution
    VkExtent2D referenceSize{};

    // render pixels per reference pixel: lod error, line width
    float renderScale = 1;
    // fixed target resolution, independent of the window: no resolve, the viewport letterboxes it
    bool fixedSize = false;
    // render pixels per window pixel: mouse position
    glm::vec2 mouseScale{1.0f};

    // typically super resolution with respect to the window size
    // 0: off - use window resolution
    // 1: off - use window resolution
    // 2: 2x resolution along width and height
    // 3: 3x resolution along width and height
    // 4: 4x resolution along width and height
    // 540:  fix render resolution to  960 x  540
    // 720:  fix render resolution to 1280 x  720
    // 1080: fix render resolution to 1920 x 1080
    // 1440: fix render resolution to 2560 x 1440
    // 2160: fix render resolution to 3840 x 2160
    // 1024: fix render resolution to square
    // 2048: fix render resolution to square
    // 4096: fix render resolution to square
    int  supersample    = 0;
    bool hbaoQuarterRes = true;

    bool useResolved         = false;
    bool useRasterization    = false;
    bool useVisibilityBuffer = false;
    bool useRasterAtomic     = false;

    // linear HDR, tonemapped into `imgColorLdr`
    VkFormat colorFormat            = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat ldrColorFormat         = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat raytracingDepthFormat  = VK_FORMAT_R32_SFLOAT;
    VkFormat visibilityBufferFormat = VK_FORMAT_R32G32_UINT;
    VkFormat depthStencilFormat;

    VkViewport  viewport;
    VkRect2D    scissor;
    VkImageView viewDepth = VK_NULL_HANDLE;

    nvvk::Image imgColor           = {};
    nvvk::Image imgColorLdr        = {};
    nvvk::Image imgColorResolved   = {};
    nvvk::Image imgDepthStencil    = {};
    nvvk::Image imgRaytracingDepth = {};
    nvvk::Image imgRasterAtomic    = {};
    // hybrid path: rasterized instance / cluster+triangle ids
    nvvk::Image imgVisibilityBuffer = {};
    nvvk::Image imgHizFar[2]        = {};

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

#if USE_DLSS
    DlssMode                    dlssMode           = DlssMode::eNone;
    bool                        hasDenoiser        = false;
    DlssDenoiser                dlssDenoiser       = {};
    DlssUpscaler                dlssUpscaler       = {};
    NVSDK_NGX_PerfQuality_Value dlssQuality        = NVSDK_NGX_PerfQuality_Value(-1);
    VkImageLayout               dlssSrColorLayout  = VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout               dlssSrMotionLayout = VK_IMAGE_LAYOUT_GENERAL;
#endif
  };

  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, const nvvk::QueueInfo& queue, const nvvk::QueueInfo& queueTransfer);
  void deinit();

  bool initFramebuffer(const VkExtent2D& windowSize, int supersample, bool hbaoQuarterRes);
  void updateFramebufferRenderSizeDependent(VkCommandBuffer cmd);
  // rebuilds render-size dependent resources so rasterization-only targets (HBAO, raster atomic)
  // are only allocated while a rasterization renderer is active
  void setFramebufferUseRasterization(bool enabled);
  void setFramebufferUseVisibilityBuffer(bool enabled);
  // the hybrid renderer's compute raster, rasterization always has the image
  void setFramebufferUseRasterAtomic(bool enabled);
#if USE_DLSS
  void updateFramebufferDlss(VkCommandBuffer cmd);
  // falls back to `eNone` if DLSS fails to initialize, or, without `m_dlssWaitForInit`, while its NGX init is still running
  void setFramebufferDlss(DlssMode mode, NVSDK_NGX_PerfQuality_Value dlssQuality);
  void setFramebufferDlss(bool enabled, NVSDK_NGX_PerfQuality_Value dlssQuality);

  // NGX init and first feature creation take several seconds, they run on a worker thread
  void initDlssAsync(DlssMode mode, NVSDK_NGX_PerfQuality_Value dlssQuality);
  bool isDlssLoading() const;
  // returns the mode whose async init just finished, eNone otherwise
  DlssMode pollDlssInit();

  bool m_dlssWaitForInit = true;
  // set when an evaluation failed, the app then disables DLSS at the next safe point
  bool m_dlssEvaluateFailed = false;
#endif
  void deinitFramebufferRenderSizeDependent();
  void deinitFramebuffer();

  void updateFramebufferMemBytes();
  void updateFramebufferScales();

  glm::vec2 getFramebufferMouseScale() const;

  void beginFrame(uint32_t cycleIndex);
  void postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void endFrame();

  // re-bakes the physical sky lighting if its parameters changed
  void cmdUpdateSkyLighting(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  // sampled by the raster and ray tracing shading, see `BINDINGS_SKY_ENV_TEX` etc.
  const VkDescriptorImageInfo& getSkyEnvDescriptor() const { return m_skyLighting.env.descriptor; }
  const VkDescriptorImageInfo& getSkyIrradianceDescriptor() const { return m_skyLighting.irradiance.descriptor; }

  // HDRI environment, an empty path or a failed load leaves a 1x1 dummy bound. Waits for the device,
  // renderers must re-create their descriptor sets afterwards.
  bool loadEnvHdri(const std::filesystem::path& file);
  bool hasEnvHdri() const { return m_envHdri.isValid(); }
  // `BINDINGS_ENV_HDRI_TEX` and `BINDINGS_ENV_HDRI_ACCEL_SSBO`
  const VkDescriptorImageInfo& getEnvHdriDescriptor() { return m_envHdri.getHdrImage().descriptor; }
  nvvk::Buffer                 getEnvHdriAccel() const { return m_envHdri.getEnvAccel(); }
  VkExtent2D                   getEnvHdriSize() const { return m_envHdri.getHdrImageSize(); }

  bool reloadShadingShaders();
  // renderers cannot run without the tonemap and sky bake pipelines
  bool hasShadingPasses() const;

  void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, uint32_t idx);
  // without `apply` the tonemap pass composites the result
  void cmdHBAO(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, bool apply = true);

  // some vulkan implementations only support 16 bit per grid component
  // need to convert the 1D intended launch into a grid.
  void cmdLinearDispatch(VkCommandBuffer cmd, uint32_t count) const
  {
    if(!count)
      return;

    if(!m_use16bitDispatch || count <= 0xFFFF)
    {
      vkCmdDispatch(cmd, count, 1, 1);
    }
    else
    {
      glm::uvec3 grid = shaderio::fit16bitLaunchGrid(count);
      assert(grid.x <= 0xFFFF && grid.y <= 0xFFFF && grid.z <= 0xFFFF);
      vkCmdDispatch(cmd, grid.x, grid.y, grid.z);
    }
  }

  void getReadbackData(shaderio::Readback& readback);
  void clearReadbackData();

  //////////////////////////////////////////////////////////////////////////

  shaderc::CompileOptions makeCompilerOptions() { return shaderc::CompileOptions(m_glslCompiler.options()); }

  // `dumpFileName` defaults to the file's stem, provide it when the same file is compiled
  // more than once, so the dumped files stay unique
  bool compileShader(shaderc::SpvCompilationResult& compiled,
                     VkShaderStageFlagBits          shader,
                     const std::filesystem::path&   filePath,
                     shaderc::CompileOptions*       options      = nullptr,
                     const std::filesystem::path&   dumpFileName = {});

  // `variant` must be provided when the same file is compiled more than once (different defines),
  // it keeps the shader's file name, and therefore the dumped files, unique
  bool compileShader(Shader&                      shader,
                     VkShaderStageFlagBits        shaderStage,
                     const std::filesystem::path& filePath,
                     shaderc::CompileOptions*     options = nullptr,
                     const char*                  variant = nullptr);

  // flags all pipelines must be created with, so that `dumpPipelineInternals` can work
  VkPipelineCreateFlags getPipelineCreateFlags() const
  {
    return m_dumpInternal ? VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR : 0;
  }

  // dumps the driver's internal representation (isa etc.) into the working directory,
  // named after the shader's file. does nothing unless `m_dumpInternal` is active.
  void dumpPipelineInternals(VkPipeline pipeline, const Shader& shader);

  // sets up compInfo's stage.pNext and flags, and dumps the pipeline internals if enabled
  void createComputePipeline(VkComputePipelineCreateInfo& compInfo, const Shader& shader, VkPipeline& pipeline);

  // tests if all shaders compiled well, returns false if not
  // also destroys all shaders if not all were successful.
  bool verifyShaders(size_t numShaders, const Shader* shaders)
  {
    for(size_t i = 0; i < numShaders; i++)
    {
      if(shaders[i].compiled.GetCompilationStatus() != shaderc_compilation_status_null_result_object
         && shaders[i].compiled.GetCompilationStatus() != shaderc_compilation_status_success)
        return false;
    }

    return true;
  }
  template <typename T>
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(Shader), (const Shader*)&container);
  }

  void destroyPipelines(size_t numPipelines, VkPipeline* pipelines)
  {
    for(size_t i = 0; i < numPipelines; i++)
    {
      vkDestroyPipeline(m_device, pipelines[i], nullptr);
      pipelines[i] = nullptr;
    }
  }
  template <typename T>
  void destroyPipelines(T& container)
  {
    destroyPipelines(sizeof(T) / sizeof(VkPipeline), (VkPipeline*)&container);
  }

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createTempCmdBuffer();
  void            tempSyncSubmit(VkCommandBuffer cmd);

  //////////////////////////////////////////////////////////////////////////

  void cmdBeginRendering(VkCommandBuffer    cmd,
                         bool               hasSecondary = false,
                         VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR);
#if USE_DLSS
  void cmdBeginRenderingDlssSrColor(VkCommandBuffer    cmd,
                                    VkAttachmentLoadOp loadOpColor = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                    VkAttachmentLoadOp loadOpDepth = VK_ATTACHMENT_LOAD_OP_CLEAR);
  void cmdBeginRenderingDlssSr(VkCommandBuffer    cmd,
                               VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                               VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                               VkAttachmentLoadOp loadOpMotion = VK_ATTACHMENT_LOAD_OP_CLEAR);
  void cmdPrepareDlssSrEvaluation(VkCommandBuffer cmd);
  bool cmdEvaluateDlssSr(VkCommandBuffer cmd, glm::vec2 jitter, nvvk::ProfilerGpuTimer& profiler);
#endif
  void cmdBeginRayTracing(VkCommandBuffer cmd);

  void cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;

  //////////////////////////////////////////////////////////////////////////

  template <typename T>
  VkResult createBufferTyped(nvvk::BufferTyped<T>&     buffer,
                             size_t                    elementCount,
                             VkBufferUsageFlagBits2    bufferUsageFlags,
                             VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                             VmaAllocationCreateFlags  vmaAllocFlags = 0,
                             VkDeviceSize              minAlignment  = 0,
                             std::span<const uint32_t> queueFamilies = {})
  {
    return m_allocator.createBuffer(buffer, elementCount * nvvk::BufferTyped<T>::value_size, bufferUsageFlags,
                                    vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }

  VkResult createBuffer(nvvk::Buffer&             buffer,
                        VkDeviceSize              bufferSize,
                        VkBufferUsageFlagBits2    bufferUsageFlags,
                        VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                        VmaAllocationCreateFlags  vmaAllocFlags = 0,
                        VkDeviceSize              minAlignment  = 0,
                        std::span<const uint32_t> queueFamilies = {})
  {
    return m_allocator.createBuffer(buffer, bufferSize, bufferUsageFlags, vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }

  VkResult createLargeBuffer(nvvk::LargeBuffer&     buffer,
                             VkDeviceSize           bufferSize,
                             VkBufferUsageFlagBits2 bufferUsageFlags,
                             VkDeviceSize           chunkSize = nvvk::ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE,
                             uint32_t               initialChunkCount = 0)
  {
    return m_allocator.createLargeBuffer(buffer, bufferSize, bufferUsageFlags, m_queue.queue, VK_NULL_HANDLE, chunkSize,
                                         VMA_MEMORY_USAGE_AUTO, {}, 0, {}, initialChunkCount);
  }

  VkResult resizeLargeBuffer(nvvk::LargeBuffer& buffer, VkDeviceSize newSize, VkFence sparseBindingFence = VK_NULL_HANDLE) const
  {
    bool bindNeeded;
    return m_allocator.resizeLargeBuffer(buffer, newSize, bindNeeded, m_queue.queue, sparseBindingFence);
  }

  VkDeviceSize getDeviceLocalHeapSize() const;

  VkDeviceSize getFramebufferMemBytes() const { return m_frameBufferMemBytes; }

  bool isBufferSizeValid(VkDeviceSize size) const;

  //////////////////////////////////////////////////////////////////////////

  void simpleUploadBuffer(const nvvk::Buffer& buffer, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, 0, buffer.bufferSize, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  void simpleUploadBuffer(const nvvk::Buffer& buffer, size_t offset, size_t sz, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, offset, sz, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

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

    void checkedFlush(size_t sz)
    {
      if(sz)
      {
        if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz))
        {
          flush();
        }
      }
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, size_t offset, size_t sz, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      if(sz)
      {
        if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz) && flushState == FlushState::ALLOW_FLUSH)
        {
          flush();
        }

        if(!m_cmd)
        {
          m_cmd = m_resources.createTempCmdBuffer();
        }
        T* mapping = nullptr;
        NVVK_CHECK(m_resources.m_uploader.appendBufferMapping(dst, offset, sz, mapping));

        if(src)
        {
          memcpy(mapping, src, sz);
        }

        return mapping;
      }
      return nullptr;
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      return uploadBuffer(dst, 0, dst.bufferSize, src, flushState);
    }

    void fillBuffer(const nvvk::Buffer& dst, uint32_t fillValue)
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      vkCmdFillBuffer(m_cmd, dst.buffer, 0, dst.bufferSize, fillValue);
    }

    void fillBuffer(const nvvk::Buffer& dst, size_t offset, size_t size, uint32_t fillValue)
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      vkCmdFillBuffer(m_cmd, dst.buffer, offset, size, fillValue);
    }

    // must call flush at end of operations
    void flush()
    {
      if(m_cmd)
      {
        m_resources.m_uploader.cmdUploadAppended(m_cmd);
        m_resources.tempSyncSubmit(m_cmd);
        m_resources.m_uploader.releaseStaging();
        m_cmd = nullptr;
      }
    }

    void abort()
    {
      m_resources.m_uploader.cancelAppended();
      m_resources.m_uploader.releaseStaging();
    }

    ~BatchedUploader() { assert(!m_cmd && "must call flush at end"); }

  private:
    Resources&      m_resources;
    VkDeviceSize    m_maxBatchSize = 0;
    VkCommandBuffer m_cmd          = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////

  static constexpr VkPipelineStageFlags2 s_supportedShaderStages =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

  VkDevice         m_device          = {};
  VkPhysicalDevice m_physicalDevice  = {};
  nvvk::QueueInfo  m_queue           = {};
  nvvk::QueueInfo  m_queueTransfer   = {};
  VkCommandPool    m_tempCommandPool = {};

  nvvk::ResourceAllocator m_allocator        = {};
  nvvk::SamplerPool       m_samplerPool      = {};
  VkSampler               m_samplerBiLinear  = {};
  VkSampler               m_samplerTriLinear = {};
  nvvkglsl::GlslCompiler  m_glslCompiler     = {};
  nvvk::StagingUploader   m_uploader         = {};
  // async transfer-queue uploader for load-time texture uploads (ownership transfer to graphics);
  // drained every frame in onRender via cmdDrainOwnershipBarriers()
  nvvk::AsyncUploader m_asyncUploader = {};

  FrameBuffer m_frameBuffer;
  struct CommonBuffers
  {
    nvvk::BufferTyped<shaderio::FrameConstants> frameConstants;
    nvvk::BufferTyped<shaderio::Readback>       readBack;
    nvvk::BufferTyped<shaderio::Readback>       readBackHost;
  } m_commonBuffers;

  nvvk::PhysicalDeviceInfo         m_physicalDeviceInfo  = {};
  VkPhysicalDeviceMemoryProperties m_memoryProperties    = {};
  nvvk::GraphicsPipelineState      m_basicGraphicsState  = {};
  uint32_t                         m_cycleIndex          = 0;
  size_t                           m_fboChangeID         = ~0;
  glm::vec4                        m_bgColor             = {0, 0, 0, 1.0};
  VkDeviceSize                     m_frameBufferMemBytes = 0;

  VkPhysicalDeviceMeshShaderPropertiesEXT m_meshShaderPropsEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  VkPhysicalDeviceMeshShaderPropertiesNV m_meshShaderPropsNV = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_NV};

  bool m_use16bitDispatch = false;
  // "force16bitdispatch", testing only: takes effect at `init` time
  bool m_force16bitDispatch             = false;
  bool m_supportsMeshShaderNV           = false;
  bool m_supportsClusterRaytracing      = false;
  bool m_supportsPartitionedTlas        = false;
  bool m_supportsBarycentrics           = false;
  bool m_supportsSmBuiltinsNV           = false;
  bool m_supportsPipelineExecutableInfo = false;
  bool m_dumpSpirv                      = false;
  bool m_dumpInternal                   = false;

  HbaoPass        m_hbaoPass;
  HbaoPass::Frame m_hbaoFrame;

  NVHizVK                       m_hiz;
  NVHizVK::Update               m_hizUpdate[2];
  shaderc::SpvCompilationResult m_hizShaders[NVHizVK::SHADER_COUNT];

  QueueStateManager m_queueStates;
  VrdxSorter        m_vrdxSorter{};

private:
  void initShadingPasses();
  void deinitShadingPasses();
  void updateTonemapDescriptors();
  void cmdTonemap(VkCommandBuffer cmd, nvvk::ProfilerGpuTimer& profiler, uint32_t variant);

  struct TonemapPass
  {
    // without HBAO, compositing HBAO
    static const uint32_t VARIANT_PLAIN = 0;
    static const uint32_t VARIANT_HBAO  = 1;
    static const uint32_t VARIANTS      = 2;

    Shader               shaders[VARIANTS];
    nvvk::DescriptorPack dset;
    VkPipelineLayout     pipelineLayout{};
    VkPipeline           pipelines[VARIANTS]{};
  } m_tonemap;

  struct SkyLightingPass
  {
    Shader               shader;
    nvvk::DescriptorPack dset;
    VkPipelineLayout     pipelineLayout{};
    VkPipeline           pipeline{};

    nvvk::Image env;
    nvvk::Image irradiance;
    VkImageView envMipViews[SKY_ENV_MIPS]{};
    VkImageView irradianceStorageView{};

    // what is currently baked
    shaderio::SkyBakePush baked{};
    bool                  valid = false;
  } m_skyLighting;

  nvvk::HdrIbl m_envHdri;
  uint32_t     m_envHdriVersion = 0;
};


}  // namespace lodclusters
