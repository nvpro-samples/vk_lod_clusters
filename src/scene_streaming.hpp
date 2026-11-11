/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scene_streaming_utils.hpp"

namespace lodclusters {
/*
  The scene is not loaded fully but streamed in.
  The lowest detail clusters (along with their CLAS) are persistently loaded,
  anything else is handled dynamically.

  Streaming requests run through four stages:

  1. On the device we fill the request task details.
     During traversal missing geometry groups are appended to the request
     load array.

     see `USE_STREAMING` within `traversal_run.comp.glsl`

     After traversal any groups that haven't been accessed in a while are 
     appended to the request unload array.
     
     see `stream_agefilter_groups.comp.glsl`

     At the end of the frame we download the request to host.

  2. The request is handled on the host after checking its availability.
     It triggers the storage upload of newly loaded geometry groups.
     It also prepares an update task, which encodes the patching of the scene.

     see `SceneStreaming::handleCompletedRequest`

  3. Once the storage upload is completed, the appropriate update task is run.
     This update task actually patches the device side buffers so the loads
     and unloads become effective.
     When ray tracing is active, we will also build the CLAS of the newly loaded
     groups and handle their allocation management along with the patching.

     see `stream_update_scene.comp.glsl`.

     CLAS allocation management is done either through a persistent
     allocator system (`stream_allocator...` shader files) or through a simple
     compaction system (`stream_compaction...` shader files).

  4. After the update task is completed on the device the host can
     safely release the memory of unloaded groups. This memory is then
     recycled when we load new geometry groups in (2).

   The streaming system has quite some configurable options, mostly balancing how 
   much operations should be done within a single frame.
   There is also the ability to use an asynchronous transfer queue for the data uploads,
   otherwise we just upload on the main queue prior the patch operations.

   Lastly another major option is how the CLAS are allocated within the
   fixed size clas buffer. Since the actual size of a CLAS is only
   known on the device after it was built and the estimates from the host
   can be a lot higher. We used solutions that can be implemented
   on the device, not relying on further host readbacks but still
   trying to make efficient use of the memory based on actual sizes.

   Two options are provided, and they both first build new CLAS into
   scratch space before moving them to their resident location.

   - Compaction: 
     This simple scheme is based on a basic compaction algorithm that 
     packs all resident cluster CLAS tightly before appending newly built ones.
     This can cause bursts of high amount of memory movement and a lot of bandwidth 
     and scratch space consumption. This is despite the fact that the new cluster
     API does provide functionality for moving objects to overlapping memory destinations.
     
     We do not recommend this, but it is the easiest to get going.

     see `stream_compaction...` shader files
   
   - Persistent Allocator:
     In this option we implement a persistent memory manager on the device
     so that clas are moved only once after initial building. The system
     builds arrays of free gaps up to the maximum clas size and then
     leverages them during the allocation process.
     This avoids the big peaks in clas memory movement of the simple
     compaction.
     
     A bit array is used to track which memory arrays are in use.
     Each bit represents a power of two multiple of the alignment size
     of a clas (at time of writing on NVIDIA hardware 128 bytes).

     During the loading phase of the geometry groups appropriate gaps for
     clas storage are found for all groups and the bits they cover are set.
     Respectively, the unloading of groups will clear the bits.
     A kernel is run after unloading to build the arrays of free gaps that
     the allocation can use.

     see `stream_allocator...` shader files

*/
class SceneStreaming
{
public:
  // pointers must stay valid during lifetime
  bool init(Resources* res, const Scene* scene, const StreamingConfig& config);

  // run prior the renderer starts referencing resources
  // if true CLAS for all clusters will be built
  bool updateClasRequired(bool state);

  // tear down, safe to call without init
  void deinit();

  // reset streaming state
  void reset();
  // clears the `groupDemand` of this frame's slice of geometries, see STREAMING_HITFRAME_DEMAND_CLEAR_FRAMES
  void cmdClearGroupDemandSlice(VkCommandBuffer cmd) const;

  // reload internal shaders
  bool reloadShaders()
  {
    deinitShadersAndPipelines();
    return initShadersAndPipelines();
  }

  struct FrameSettings
  {
    bool                                 useBlasCaching        = false;
    bool                                 useDiscreteLod        = false;
    float                                unloadThreshold       = 0.0f;
    bool                                 useHitStreaming       = false;
    bool                                 useHitStreamingLod    = false;
    float                                hitLodWidthFactor     = 1.0f;
    bool                                 freeze                = false;
    uint32_t                             blasCacheAgeThreshold = 16;
    uint32_t                             blasCacheMaxClusters  = 0;
    uint32_t                             blasCacheMaxBuilds    = 0;
    uint32_t                             blasCacheMinLevel     = 0;
    VkBuildAccelerationStructureFlagsKHR blasCacheFlags        = 0;
    uint32_t                             ageThreshold          = 16;
    // Frames a replaced cached blas allocation is held before being returned to the
    // allocator. The ptlas partition early-out only refreshes an instance when its
    // partition is processed, so an allocation must outlive the worst case gap between
    // two visits or a skipped instance ends up pointing at reused memory. 0 frees at once,
    // which is what the non early-out path wants.
    uint32_t blasCacheDeferFrames = 0;
    // Refuse to unload the groups a cached blas was built from. Without this they can age out
    // from under a blas that skipped instances are still tracing, which is the one case
    // `blasCacheDeferFrames` cannot cover - holding the allocation does not help once its
    // cluster references are gone. Only wanted with the early-out, otherwise it just holds
    // residency nothing needs.
    bool blasCachePinGroups = false;
    // Step a pinned cached blas down to a coarser level rather than pinning it forever. The
    // pin alone is a closed loop: the groups cannot age out, so the blas is never rebuilt, so
    // the pin never lifts. See `appendCachedBlasRetires`.
    bool blasCacheRetire = false;
  };

  // called by render thread
  // render thread must take care of barriers prior/after these operations
  // triggers main setup, ensures data is uploaded, unloaded etc.
  //    implicitly does "handleCompletedUpdate" and "handleCompletedStorage"
  //    explicitly calls `handleCompletedRequest`
  // barriers: none
  void cmdBeginFrame(VkCommandBuffer         cmd,
                     QueueState&             cmdQueueState,
                     QueueState&             asyncQueueState,
                     const FrameSettings&    settings,
                     nvvk::ProfilerGpuTimer& profiler);

  // triggers scene update & clas build if required
  // barriers: requires transfers from `cmdBeginFrame` to be completed
  void cmdPreTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, nvvk::ProfilerGpuTimer& profiler);

  // triggers age filter & clas move
  // barriers: requires compute shader writes from `cmdPreTraversal` to be completed,
  //           for ray tracing also requires acceleration structure builds
  void cmdPostTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, bool runAgeFilter, nvvk::ProfilerGpuTimer& profiler);

  // triggers request download
  // barriers: requires compute shader writes from `cmdPostTraversal` to be completed
  void cmdEndFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, nvvk::ProfilerGpuTimer& profiler);

  // statistics on streaming operations
  void getStats(StreamingStats& stats) const;

  // per lod-level residency of the persistent geometries, aggregated over all geometries
  const StreamingResidentStats& getResidentStats() const { return m_residentStats; }

  // scratch space is provided by renderer so it can alias/reuse the memory for other operations
  // this is the required size
  size_t getRequiredClasScratchSize() const { return m_clasScratchTotalSize; }
  size_t getRequiredClasScratchAlginment() const { return m_clasScratchAlignment; }

  // used by renderer
  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }
  // valid from `initClas`, every geometry keeps its lowest detail group resident
  uint64_t getLowDetailBlasAddress(size_t geometryID) const
  {
    return m_shaderGeometries[geometryID].lowDetailBlasAddress;
  }
  const nvvk::Buffer&             getShaderStreamingBuffer() const { return m_shaderBuffer; }
  const shaderio::SceneStreaming& getShaderStreamingData() const { return m_shaderData; }
  const StreamingConfig&          getStreamingConfig() const { return m_config; }
  uint32_t                        getTotalActiveGroups() const { return m_totalActiveGroups; }
  VkDeviceAddress                 getGeometryLevelOffsetsAddress() const { return m_geometryLevelOffsets.address; }

  // device side memory usage, reserved or current state
  size_t getClasSize(bool reserved) const;
  size_t getBlasSize(bool reserved) const;
  size_t getGeometrySize(bool reserved) const;
  size_t getOperationsSize() const { return m_operationsSize + m_clasOperationsSize; }

  float getLoadFactor() const;

  uint32_t getMaxCachedBlasBuilds() const { return m_updates.getMaxCachedBlasBuilds(); }
  // Why cached blas entries were repatched this frame. Only `higherDetail` can be deferred:
  // the other two fire because the groups the old blas was built from are already gone.
  struct CachedBlasPatchCounts
  {
    uint32_t lowerDetail    = 0;
    uint32_t higherDetail   = 0;
    uint32_t invalidateOnly = 0;
    uint32_t deferred       = 0;  // old allocation held rather than returned
    uint32_t unsafeFreed    = 0;  // old allocation returned at once, forces a full revalidation
    uint32_t retired        = 0;  // pinned entry given up so its groups can age out
  };
  const CachedBlasPatchCounts& getFramePatchCounts() const { return m_framePatchCounts; }
  uint32_t                     getFramePinnedUnloads() const { return m_framePinnedUnloads; }
  uint32_t                     getFramePinnedDeferredUnloads() const { return m_framePinnedDeferredUnloads; }

  // whether this group backs a cached blas and must stay resident, and which one: `Deferred`
  // is a blas already replaced but still held, i.e. a group only the deferral window protects
  enum CachedBlasPin
  {
    CACHED_BLAS_PIN_NONE,
    CACHED_BLAS_PIN_LIVE,
    CACHED_BLAS_PIN_DEFERRED,
  };
  CachedBlasPin cachedBlasGroupPin(const GeometryGroup& geometryGroup) const;

  // Cached blas allocations that were returned to the allocator without waiting out
  // `blasCacheDeferFrames`, so something skipped may still point at them. Non zero means the
  // early-out has to reprocess every partition this frame. With deferral enabled and
  // within budget this stays zero, which is the whole point - a patch alone must not cost a
  // full revalidation, it happens on nearly every frame the camera moves.
  uint32_t consumeUnsafeCachedBlasFree()
  {
    uint32_t count              = m_frameUnsafeCachedBlasFree;
    m_frameUnsafeCachedBlasFree = 0;
    return count;
  }

  void updateBindings(const nvvk::Buffer& sceneBuildingBuffer);

  void resetCachedBlas(Resources::BatchedUploader& uploader);
  void resetCachedBlas();

  // for debugging one can limit the amount of streaming requests to perform
  //  < 0 means all requests are handled (regular usage)
  // == 0 stops handling requests
  //  > 0 does as many requests until decremented to zero
#ifndef NDEBUG
  static const int32_t s_defaultDebugFrameLimit = -1;
#else
  static const int32_t s_defaultDebugFrameLimit = -1;
#endif
  int32_t m_debugFrameLimit = s_defaultDebugFrameLimit;

private:
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  StreamingConfig m_config;
  bool            m_requiresClas;
  size_t          m_persistentGeometrySize;
  size_t          m_operationsSize;
  size_t          m_clasOperationsSize;
  size_t          m_blasSize;
  size_t          m_peakGeometrySize = 0;
  uint32_t        m_peakFrameIndex   = ~0;
  uint32_t        m_lastUpdateIndex;
  uint32_t        m_frameIndex;
  StreamingStats  m_stats;

  // When growing the clas storage we may actually have more allocations left
  // than a request's readback reports, given requests recorded prior to the grow
  // may still be in flight.
  // The delta accumulates grows and stays active until we pop a request that was
  // recorded at or after the frame of the last grow, as only then the readback
  // reflects the grown state.
  uint32_t m_clasGrowMaxSizedLeftDelta = 0;
  uint32_t m_clasGrowFrameIndex        = 0;

  // persistent scene data

  struct PersistentGeometry
  {
    nvvk::BufferTyped<shaderio::Node>             nodes;
    nvvk::BufferTyped<shaderio::BBox>             nodeBboxes;
    nvvk::BufferTyped<shaderio::LodLevel>         lodLevels;
    nvvk::BufferTyped<uint64_t>                   groupAddresses;
    nvvk::BufferTyped<shaderio::GroupOverlapInfo> groupOverlapInfos;
    nvvk::BufferTyped<uint32_t>                   groupOverlapEntries;
    nvvk::BufferTyped<uint32_t>                   groupDemand;
    nvvk::Buffer                                  lowDetailGroupsData;
    uint32_t                                      lodLevelsCount                                = 0;
    uint32_t                                      lodLoadedGroupsCount[SHADERIO_MAX_LOD_LEVELS] = {};
    uint32_t                                      lodGroupsCount[SHADERIO_MAX_LOD_LEVELS]       = {};
    uint32_t                                      cachedBlasUpdateFrame                         = 0;
    uint32_t                                      cachedBlasLevel = TRAVERSAL_INVALID_LOD_LEVEL;
    // last published `shaderio::Geometry::discreteLodLevel` for USE_DISCRETE_LOD
    uint32_t                  discreteLodLevel     = TRAVERSAL_INVALID_LOD_LEVEL;
    nvvk::BufferSubAllocation cachedBlasAllocation = {};
    // lod levels a deferred cached blas was built from, one bit each. Those groups stay
    // pinned as well, see `cachedBlasGroupPin`.
    uint32_t deferredBlasLevelMask = 0;
    // the age filter asked for a group this cached blas is built from, so the entry is
    // queued for retirement, see `appendCachedBlasRetires`
    bool cachedBlasRetirePending = false;
  };

  std::vector<PersistentGeometry>       m_persistentGeometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;
  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;

  // CPU decode scratch, reused across group decompression (single-threaded)
  std::vector<uint32_t> m_decompressScratch;

  // faster stream-in: compact level-major table [reverseLevel*numGeometries + geometryIndex]
  // of per-reverse-lod-level flat prefix offsets, binary-searched by the level-wise demand
  // pass. Kept out of shaderio::Geometry (that struct is copied per traversal thread); the
  // demand shader receives its address via push constant.
  nvvk::BufferTyped<uint32_t> m_geometryLevelOffsets;

  // faster stream-in: total groups across all active geometries in the flat
  // global-group space (prefix sum), sizes the single stream_hit_demand_tree dispatch
  uint32_t m_totalActiveGroups = 0;

  // per lod-level residency of the persistent geometries, maintained incrementally
  // during loads/unloads (see `getResidentStats`)
  StreamingResidentStats m_residentStats;

  void initGeometries(Resources& res, const Scene* scene);
  void resetGeometryGroupAddresses(Resources::BatchedUploader& uploader);

  // streaming system related

  // the main buffer and its content used to provide data to all compute kernels,
  // it is updated within `cmdBeginFrame`
  shaderio::SceneStreaming m_shaderData;
  nvvk::Buffer             m_shaderBuffer;

  // streaming requests run through 3 stages:
  // first a request is tasked (what geometry groups to load or unload)
  StreamingTaskQueue m_requestsTaskQueue;
  // when a request is handled it will trigger an upload of newly requested geometry data
  // it also prepares an update task to the scene
  StreamingTaskQueue m_storageTaskQueue;
  // once the storage transfer has completed an update task is run to configure
  // the scene's state making the changes of loaded and unloaded groups effective.
  StreamingTaskQueue m_updatesTaskQueue;

  // each frame we prepare a request of what geometry groups to load or unload
  StreamingRequests m_requests;
  // manages the table of resident cluster groups
  StreamingResident m_resident;
  // manages the clas memory of resident cluster groups if `usePersistentClasAllocator` was
  // set in the options
  StreamingAllocator m_clasAllocator;
  // manages the upload and storage of the cluster group geometry data
  StreamingStorage m_storage;
  // manages updates to the scene, this is where loads/unloads are becoming effective
  StreamingUpdates m_updates;

  // manages memory of cached BLAS builds
  nvvk::BufferSubAllocator m_cachedBlasAllocator;
  // GEOMETRY_CACHED_BLAS_STRIDE entries per geometry, addresses of the cached blas
  nvvk::BufferTyped<uint64_t> m_cachedBlasAddressesBuffer;
  uint32_t                    m_cachedBlasAlignment = 4;

  // ui convenience only, not part of the caching algorithm, see `appendBlasCacheRevalidation`.
  // Cursor of the sweep that drops cached BLAS which a lowered `blasCacheMinLevel` excludes,
  // ~0 is idle.
  uint32_t m_blasCacheMinLevelLast   = 0;
  uint32_t m_blasCacheRevalidateNext = ~0u;
  uint32_t m_cachedBlasCount         = 0;
  // cached blas freed without the deferral guarantee, since the last consume
  uint32_t m_frameUnsafeCachedBlasFree = 0;
  // the same per update task, they count once the task is applied, which with decoupled
  // transfers is frames after its request
  uint32_t m_updateUnsafeCachedBlasFree[STREAMING_MAX_ACTIVE_TASKS] = {};
  // update task of the request being handled, INVALID_TASK_INDEX outside of it
  uint32_t m_requestUpdateIndex = INVALID_TASK_INDEX;
  // this frame's patch breakdown, reset in `cmdBeginFrame`
  CachedBlasPatchCounts m_framePatchCounts;

  // A replaced cached blas is held here for `blasCacheDeferFrames` after `startFrame`.
  // The current value is used, it follows the revalidation window when that changes.
  struct DeferredCachedBlasFree
  {
    nvvk::BufferSubAllocation allocation;
    // frame its update task was applied, ~0 while still pending
    uint32_t startFrame  = 0;
    uint32_t updateIndex = INVALID_TASK_INDEX;
    uint32_t geometryID  = 0;
    uint32_t lodLevel    = 0;
  };
  std::vector<DeferredCachedBlasFree> m_deferredCachedBlasFrees;
  // cap on how much may sit in the queue, beyond it we free early and flag it
  size_t m_deferredCachedBlasMax = 0;
  // geometries whose live cached blas the age filter wants the groups of
  std::vector<uint32_t> m_cachedBlasRetires;
  // unloads refused this frame because a cached blas is built from that group
  uint32_t m_framePinnedUnloads = 0;
  // subset of those held only by a deferred cached blas, not by the live one
  uint32_t m_framePinnedDeferredUnloads = 0;

  // `subFree` unless the caller asked for deferral and the blas is still usable, in which case
  // it is queued. Clears the caller's handle either way, same contract as `subFree`, so it
  // cannot be freed twice. `stillBacked` must be false whenever the groups the old blas was
  // built from are no longer resident - holding the allocation does not help then, its cluster
  // references are already dangling, and the early-out has to be told.
  void deferCachedBlasFree(uint32_t geometryID, PersistentGeometry& geometry, const FrameSettings& settings, bool stillBacked);
  // return everything whose hold has expired
  void releaseDeferredCachedBlasFrees(const FrameSettings& settings);
  void flagUnsafeCachedBlasFree();

  // This is the main function where we react on streaming requests.
  // It processes the request by issuing new storage upload work and prepares the scene patching and resident update task.
  // returns the updateTaskIndex to be handled immediately in this frame if it's != INVALID_TASK_INDEX
  uint32_t handleCompletedRequest(VkCommandBuffer      cmd,
                                  QueueState&          cmdQueueState,
                                  QueueState&          asyncQueueState,
                                  const FrameSettings& settings,
                                  uint32_t             popRequestIndex);

private:
  void handleBlasCaching(StreamingUpdates::TaskInfo& updateTask, const FrameSettings& settings);
  // rasterization: publishes the lod level from which on all levels are fully resident
  void handleDiscreteLod(StreamingUpdates::TaskInfo& updateTask);
  // ui convenience only, see definition
  void appendBlasCacheRevalidation(StreamingUpdates::TaskInfo& updateTask, const FrameSettings& settings);
  void appendCachedBlasRetires(StreamingUpdates::TaskInfo& updateTask,
                               const FrameSettings&        settings,
                               uint32_t&                   writeIndex,
                               uint32_t&                   cachedBuildsTotal,
                               uint32_t&                   cachedClustersTotal);

  bool allocateCachedBlas(const PersistentGeometry&  geometry,
                          uint32_t                   lodClustersCount,
                          const FrameSettings&       settings,
                          nvvk::BufferSubAllocation& subAllocation);

  // ray tracing specific
  VkClusterAccelerationStructureTriangleClusterInputNV m_clasTriangleInput;

  nvvk::Buffer m_clasLowDetailBuffer;
  size_t       m_clasLowDetailSize;

  nvvk::Buffer m_clasLowDetailBlasBuffer;

  // max size of a clas can have
  size_t m_clasSingleMaxSize;
  // max storage size of all newly clas built in a frame
  size_t m_clasScratchNewClasSize;
  // scratch space for building the new clas
  size_t m_clasScratchNewBuildSize;
  // scratch space for moving clas
  size_t m_clasScratchMoveSize;
  // total scratch space needed in a frame
  size_t m_clasScratchTotalSize;
  // alignment of scratch operations
  size_t m_clasScratchAlignment;

  bool initClas();
  void deinitClas();

  // command buffer may record filling allocation manager for new empty memory space
  uint32_t tryGrowClas(VkCommandBuffer cmd, const StreamingRequests::TaskInfo& request);

  // shaders & pipelines

  struct Shaders
  {
    Shader computeAgeFilterGroups;
    Shader computeUpdateSceneRaster;
    Shader computeUpdateSceneRay;
    Shader computeUpdateClasGeometryIndices;
    Shader computeSetup;

    // if usePersistentClasAllocator
    Shader computeAllocatorBuildFreeGaps;
    Shader computeAllocatorFreeGapsInsert;
    Shader computeAllocatorSetupInsertion;
    Shader computeAllocatorUnloadGroups;
    Shader computeAllocatorLoadGroups;
    // else
    Shader computeCompactionClasOld;
    Shader computeCompactionClasNew;
  };

  struct Pipelines
  {
    VkPipeline computeAgeFilterGroups           = nullptr;
    VkPipeline computeUpdateSceneRaster         = nullptr;
    VkPipeline computeUpdateSceneRay            = nullptr;
    VkPipeline computeUpdateClasGeometryIndices = nullptr;
    VkPipeline computeSetup                     = nullptr;

    // if usePersistentClasAllocator
    VkPipeline computeAllocatorBuildFreeGaps  = nullptr;
    VkPipeline computeAllocatorFreeGapsInsert = nullptr;
    VkPipeline computeAllocatorSetupInsertion = nullptr;
    VkPipeline computeAllocatorUnloadGroups   = nullptr;
    VkPipeline computeAllocatorLoadGroups     = nullptr;
    // else
    VkPipeline computeCompactionClasOld = nullptr;
    VkPipeline computeCompactionClasNew = nullptr;
  };

  Shaders              m_shaders;
  Pipelines            m_pipelines;
  VkPipelineLayout     m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;

  bool initShadersAndPipelines();
  void deinitShadersAndPipelines();
};
}  // namespace lodclusters
