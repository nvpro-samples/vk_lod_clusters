/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <memory>

#include <nvvk/acceleration_structures.hpp>
#include <nvvk/compute_pipeline.hpp>

#include "resources.hpp"
#include "scene.hpp"
#include "scene_preloaded.hpp"
#include "scene_streaming.hpp"
#include "scene_textures.hpp"

namespace lodclusters {

// There are two implementations for a renderable scene.
// Everything is preloaded or we stream in data dynamically.
class RenderScene
{
public:
  const Scene*   scene        = nullptr;
  bool           useStreaming = false;
  ScenePreloaded scenePreloaded;
  SceneStreaming sceneStreaming;
  SceneTextures  sceneTextures;

  // pointers must stay valid during lifetime
  bool init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_, const SceneTexturesConfig& texturesConfig);

  // Two-phase init for asynchronous loading: initTextures() is safe to call from a background
  // loader thread (uploads go through the transfer queue), while initGeometry() does the GPU
  // geometry setup (CLAS/BLAS build) on the primary queue and must run on the main thread.
  // init() simply runs both.
  bool initTextures(Resources* res, const Scene* scene_, const SceneTexturesConfig& texturesConfig, const SceneProgressInfo& progressInfo = {});
  bool initGeometry(Resources* res, const StreamingConfig& streamingConfig_, bool useStreaming_);

  void deinit();

  void streamingReset();

  bool updateClasRequired(bool state);

  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const;
  size_t                                       getClasSize(bool reserved) const;
  size_t                                       getBlasSize(bool reserved) const;
  size_t                                       getOperationsSize() const;
  size_t                                       getGeometrySize(bool reserved) const;
};

struct RendererConfig
{
  bool flipWinding       = false;
  bool forceTwoSided     = false;
  bool useCulledGroupLod = false;
  // hybrid path: trace the primary ray too and log an error where it disagrees with the raster
  bool useHybridMatchCheck   = false;
  bool useSorting            = false;
  bool useRenderStats        = false;
  bool useCulling            = true;
  bool useTwoPassCulling     = false;
  bool useTwoPassRejectLists = true;
  bool useBlasSharing        = true;
  bool useBlasMerging        = true;
  bool useBlasCaching        = true;
  bool useDiscreteLod        = false;
  // groups the instances spatially, which the early-out and the ptlas both consume. Neither
  // does anything without it.
  bool useBlasDedup  = false;
  bool usePartitions = false;
  bool usePtlas      = false;
  // `initRayTracingPtlasInstances` takes these from the geometry's k-DOP hull, which spans
  // every lod level.
  bool usePtlasExplicitAabb = false;
  // skip the per-instance kernels for partitions whose instances all sit on a persistent
  // blas and whose lod situation has not moved. Independent of the ptlas.
  bool usePartitionEarlyOut = true;
  // how the instances are grouped into partitions, see ScenePartitionConfig::Layout
  uint32_t partitionLayout = 1;
  // fixed grid: world units per cell, takes precedence over `partitionGridCellsPerAxis` when non zero
  float partitionGridCellSize = 0;
  // fixed grid: shifts the world origin the cells are aligned to
  glm::vec3 partitionGridOrigin = {0, 0, 0};
  // fixed grid: cells along each axis over the scene bounds
  uint32_t partitionGridCellsPerAxis = 16;
  // derived layouts: instances a partition should hold, the bvh and quadtree leaf size or the initial grid resolution
  uint32_t partitionDerivedTarget = 64;
  // derived grid: hard cap for a single partition, the resolution is refined until it fits
  uint32_t partitionDerivedMaxInstances = 1024;
  // derived layouts: budget for the number of partitions, 0 only respects the device limit. Both
  // the ptlas build cost and the trace cost track this rather than occupancy. The grid can only
  // reach it approximately, since its derivation doubles `perAxis`; the bvh layouts hit it exactly, the
  // quadtree to within two, a split adds up to three.
  uint32_t partitionDerivedMaxCount = 4096;
  bool     useShading               = true;
  bool     useDebugVisualization    = true;
  bool     useEXTmeshShader         = false;
  bool     useComputeRaster         = true;
  bool     usePrimitiveCulling      = false;
  bool     useDepthOnly             = false;
  bool     usePersistentTraversal   = true;
  bool     useVulkanMemoryModel     = false;
  bool     useAtomicLoadStore       = true;
  bool     usePersistentHitDemand   = true;
  int      textureLodMode           = 0;
  // hit-streaming demand propagation strategy (see shaderio.h STREAM_PROPAGATION_*,
  // docs/hitbased_streaming.md): 0 = resident levels (1 hop), 1 = resident subtree (BFS),
  // 2 = all levels (level-wise sweep), 3 = all levels rolling (sweep, one level/frame).
  // Recompiles shaders on change.
  int streamPropagationMode = 1;

  // basic path tracing: shading done in the ray-generation shader, 1 spp,
  // relies on DLSS Ray Reconstruction for denoising (noisy when DLSS is off).
  bool usePathtrace = false;
  // image generation as a compute shader with ray queries instead of the ray tracing pipeline
  bool useRayQuery = false;

  bool useHitStreaming = false;
  // additionally clamp the lod level based on the ray cone widths of hits,
  // requires useHitStreaming, see docs/hitbased_streaming.md
  bool useHitStreamingLod = true;
  // path depth up to which sun shadow rays register their hits too: 1 from the primary
  // hits, 2 also from the first bounce. They run a closest-hit instead of skipping it.
  int hitStreamingShadowDepth = 1;

  bool useDlss = false;
#if USE_DLSS
  NVSDK_NGX_PerfQuality_Value dlssQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
#else
  int dlssQuality = 0;
#endif

  // the maximum number of renderable clusters per frame in bits i.e. (1 << number)
  uint32_t numRenderClusterBits = 20;
  // the maximum number of traversal intermediate tasks
  uint32_t numTraversalTaskBits = 20;

  // build flags for the cluster BLAS
  VkBuildAccelerationStructureFlagsKHR clusterBlasFlags = 0;

  // cached blas lod levels per geometry, a chain from the coarsest tail level up
  uint32_t cachingLevels = 1;
  // hybrid path: own defaults for discrete lod in the caching tail, the cached levels and blas dedup
  bool     useHybridDiscreteLod = true;
  uint32_t hybridCachingLevels  = 4;
  bool     useHybridBlasDedup   = true;
};

class Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer() {};  // Defined only so that inherited classes also have virtual destructors. Use deinit().
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) { updateBasicDescriptors(res, rscene); };

  struct ResourceUsageInfo
  {
    size_t rtTlasMemBytes{};
    size_t rtBlasMemBytes{};
    size_t rtClasMemBytes{};
    size_t operationsMemBytes{};
    size_t geometryMemBytes{};

    void add(const ResourceUsageInfo& other)
    {
      rtTlasMemBytes += other.rtTlasMemBytes;
      rtBlasMemBytes += other.rtBlasMemBytes;
      rtClasMemBytes += other.rtClasMemBytes;
      operationsMemBytes += other.operationsMemBytes;
      geometryMemBytes += other.geometryMemBytes;
    }
    size_t getTotalSum() const
    {
      return rtTlasMemBytes + rtBlasMemBytes + rtClasMemBytes + geometryMemBytes + operationsMemBytes;
    }
  };

  // The readback counters are single-frame samples and swing wildly - a forced revalidation
  // frame reads completely differently from the frame next to it. Accumulated by the caller
  // over exactly the frames the profiler is averaging, so a benchmark step can report the
  // mean over the step rather than whatever the last frame happened to be.
  struct FrameStats
  {
    uint64_t frames               = 0;
    uint64_t traversalTasks       = 0;
    uint64_t renderClusters       = 0;
    uint64_t blasBuilds           = 0;
    uint64_t ptlasUpdates         = 0;
    uint64_t ptlasUpdatesPeak     = 0;
    uint64_t ptlasDirtyPartitions = 0;
    uint64_t ptlasDirtyInstances  = 0;
    uint64_t activePartitions     = 0;
    uint64_t activeInstances      = 0;
    // times the profiler threw its averages away mid step (section set changed)
    uint64_t profilerResets = 0;
    // frames that had to process every partition anyway, see `partitionInvalidateAll`
    uint64_t partitionInvalidateAllCount = 0;
    // cached blas repatch reasons, see SceneStreaming::CachedBlasPatchCounts
    uint64_t patchLowerDetail    = 0;
    uint64_t patchHigherDetail   = 0;
    uint64_t patchInvalidateOnly = 0;
    uint64_t patchDeferred       = 0;
    uint64_t patchUnsafeFreed    = 0;
    uint64_t patchRetired        = 0;
    // group unloads refused because a cached blas is built from them
    uint64_t pinnedUnloads = 0;
    // subset held only by a cached blas already replaced but still within its deferral window
    uint64_t pinnedDeferredUnloads = 0;
    // instances whose filtered hit lod scale changed
    uint64_t instanceHitScaleChanges = 0;

    uint64_t mean(uint64_t sum) const { return frames ? (sum + frames / 2) / frames : 0; }

    FrameStats& operator+=(const FrameStats& other)
    {
      frames += other.frames;
      traversalTasks += other.traversalTasks;
      renderClusters += other.renderClusters;
      blasBuilds += other.blasBuilds;
      ptlasUpdates += other.ptlasUpdates;
      ptlasUpdatesPeak = std::max(ptlasUpdatesPeak, other.ptlasUpdatesPeak);
      ptlasDirtyPartitions += other.ptlasDirtyPartitions;
      ptlasDirtyInstances += other.ptlasDirtyInstances;
      activePartitions += other.activePartitions;
      activeInstances += other.activeInstances;
      profilerResets += other.profilerResets;
      partitionInvalidateAllCount += other.partitionInvalidateAllCount;
      patchLowerDetail += other.patchLowerDetail;
      patchHigherDetail += other.patchHigherDetail;
      patchInvalidateOnly += other.patchInvalidateOnly;
      patchDeferred += other.patchDeferred;
      patchUnsafeFreed += other.patchUnsafeFreed;
      patchRetired += other.patchRetired;
      pinnedUnloads += other.pinnedUnloads;
      pinnedDeferredUnloads += other.pinnedDeferredUnloads;
      instanceHitScaleChanges += other.instanceHitScaleChanges;
      return *this;
    }
  };

  inline const FrameStats& getFrameStats() const { return m_frameStats; }
  inline void              resetFrameStats() { m_frameStats = {}; }
  inline FrameStats&       getFrameStatsRW() { return m_frameStats; }

  inline ResourceUsageInfo getResourceUsage(bool reserved) const
  {
    return reserved ? m_resourceReservedUsage : m_resourceActualUsage;
  };

  uint32_t getMaxRenderClusters() const { return m_maxRenderClusters; }
  uint32_t getMaxTraversalTasks() const { return m_maxTraversalTasks; }
  uint32_t getMaxBlasBuilds() const { return m_maxBlasBuilds; }
  // the settings after `init` dropped what the device or scene cannot do
  const RendererConfig& getConfig() const { return m_config; }

  uint32_t getOriginalMaterialID(uint32_t renderMaterialID) const
  {
    if(renderMaterialID >= m_renderMaterials.size())
      return ~0;
    return m_renderMaterials[renderMaterialID].originalID;
  }

  inline float getLodError() const { return m_lodPixelError; }

protected:
  void initBasics(Resources& res, RenderScene& rscene);
  void deinitBasics(Resources& res);

  bool initBasicShaders(Resources& res, RenderScene& rscene, bool isRaster);
  void initBasicPipelines(Resources& res, RenderScene& rscene);
  void updateBasicDescriptors(Resources& res, RenderScene& scene, const nvvk::Buffer* sceneBuildBuffer = nullptr);

  void writeRayTracingDepthBuffer(VkCommandBuffer cmd);
  void writeBackgroundSky(VkCommandBuffer cmd);
  void renderInstanceBboxes(VkCommandBuffer cmd);
  void renderClusterBboxes(VkCommandBuffer cmd, nvvk::Buffer sceneBuildBuffer, bool alpha);

  float updateLodPixelError(Resources& res, RenderScene& rscene, const FrameConfig& frame);

  struct BasicShaders
  {
    Shader fullScreenVertexShader;
    Shader fullScreenWriteDepthFragShader;
    Shader fullScreenBackgroundFragShader;

    Shader renderInstanceBboxesFragmentShader;
    Shader renderInstanceBboxesMeshShader;

    Shader renderClusterBboxesMeshShader;
    Shader renderClusterBboxesFragmentShader;
  };

  struct BasicPipelines
  {
    VkPipeline writeDepth{};
    VkPipeline background{};
    VkPipeline renderInstanceBboxes{};
    VkPipeline renderClusterBboxes{};
  };

  RendererConfig m_config;
  bool           m_isRaster                = {false};
  uint32_t       m_maxRenderClusters       = 0;
  uint32_t       m_maxTraversalTasks       = 0;
  uint32_t       m_maxBlasBuilds           = 0;
  uint32_t       m_meshShaderWorkgroupSize = 0;
  uint32_t       m_meshShaderBoxes         = 0;
  uint32_t       m_frameIndex              = 0;
  float          m_lodPixelError{};
  float          m_smoothedLoadFactor{};

  BasicShaders   m_basicShaders;
  BasicPipelines m_basicPipelines;

  std::vector<shaderio::RenderInstance> m_renderInstances;
  nvvk::Buffer                          m_renderInstanceBuffer;

  std::vector<shaderio::RenderMaterial> m_renderMaterials;
  nvvk::Buffer                          m_renderMaterialBuffer;

  ResourceUsageInfo m_resourceReservedUsage{};
  ResourceUsageInfo m_resourceActualUsage{};

  FrameStats m_frameStats{};

  nvvk::DescriptorPack m_basicDset;
  VkShaderStageFlags   m_basicShaderFlags{};
  VkPipelineLayout     m_basicPipelineLayout{};

  nvvk::Buffer m_sortingAuxBuffer;
};

//////////////////////////////////////////////////////////////////////////

// cluster rasterization shared by the raster and the hybrid renderer

// one indirect mesh task draw, `offsetEXT` / `offsetNV` point at the command in `buffer`
void cmdDrawMeshTasksIndirect(VkCommandBuffer cmd, bool useEXTmeshShader, VkBuffer buffer, VkDeviceSize offsetEXT, VkDeviceSize offsetNV);

// hiz the next cull pass tests against: with two passes pass 0 builds the second pass' hiz,
// the last pass next frame's
void cmdBuildPassHiz(VkCommandBuffer cmd, Resources& res, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, bool twoPass, uint32_t pass);

//////////////////////////////////////////////////////////////////////////

std::unique_ptr<Renderer> makeRendererRasterClustersLod();
std::unique_ptr<Renderer> makeRendererRayTraceClustersLod();
// path tracing from rasterized primary visibility, see docs/raster_hybrid.md
std::unique_ptr<Renderer> makeRendererHybridClustersLod();

}  // namespace lodclusters
