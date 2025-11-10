/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <volk.h>
#include <fmt/format.h>

#include "scene_streaming.hpp"

#define STREAMING_DEBUG_FORCE_REQUESTS 0

namespace lodclusters {

static_assert(sizeof(shaderio::ClasBuildInfo) == sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV));

template <class T>
struct OffsetOrPointer
{
  union
  {
    uint64_t offset;
    T*       pointer;
  };
};

bool SceneStreaming::init(Resources* resources, const Scene* scene, const StreamingConfig& config)
{
  assert(!m_resources && "no init before deinit");
  assert(resources && scene);
  Resources& res = *resources;

  m_resources = resources;
  m_scene     = scene;
  m_config    = config;

  m_shaderData = {};
  m_shaders    = {};
  m_pipelines  = {};

  m_requiresClas            = false;
  m_lastUpdateIndex         = 0;
  m_frameIndex              = 1;  // intentionally start at 1
  m_operationsSize          = 0;
  m_persistentGeometrySize  = 0;
  m_blasSize                = 0;
  m_clasOperationsSize      = 0;
  m_clasLowDetailSize       = 0;
  m_clasSingleMaxSize       = 0;
  m_clasScratchNewClasSize  = 0;
  m_clasScratchNewBuildSize = 0;
  m_clasScratchMoveSize     = 0;
  m_clasScratchTotalSize    = 0;
  m_stats                   = {};

  // some adjustments are required to make the config compatible
  // need at least all lo-res groups of all geometries
  m_config.maxGroups = std::max(m_config.maxGroups, uint32_t(scene->getActiveGeometryCount()));
  if(m_config.maxClusters == 0)
  {
    m_config.maxClusters = config.maxGroups * scene->m_config.clusterGroupSize;
  }
  m_config.maxClusters =
      std::max(m_config.maxClusters, uint32_t(scene->getActiveGeometryCount()) * scene->m_config.clusterGroupSize);

  m_stats.maxLoadCount     = m_config.maxPerFrameLoadRequests;
  m_stats.maxUnloadCount   = m_config.maxPerFrameUnloadRequests;
  m_stats.maxGroups        = m_config.maxGroups;
  m_stats.maxClusters      = m_config.maxClusters;
  m_stats.maxTransferBytes = m_config.maxTransferMegaBytes * 1024 * 1024;

  // setup descriptor set container
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_dsetPack.init(bindings, res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()},
                               {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)}});
  }

  if(!initShadersAndPipelines())
  {
    m_dsetPack.deinit();
    return false;
  }

  uint32_t groupCountAlignment = std::max(std::max(STREAM_AGEFILTER_GROUPS_WORKGROUP, STREAM_UPDATE_SCENE_WORKGROUP),
                                          STREAM_COMPACTION_OLD_CLAS_WORKGROUP);

  uint32_t clusterCountAlignment = STREAM_COMPACTION_NEW_CLAS_WORKGROUP;

  // setup streaming management
  m_requestsTaskQueue = {};
  m_updatesTaskQueue  = {};
  m_storageTaskQueue  = {};

  m_requests.init(res, m_config, groupCountAlignment, clusterCountAlignment);
  m_resident.init(res, m_config, groupCountAlignment, clusterCountAlignment);
  m_updates.init(res, m_config, uint32_t(m_scene->getActiveGeometryCount()), groupCountAlignment, clusterCountAlignment);
  m_storage.init(res, m_config);

  // storage uses block allocator, max may be less than what we asked for
  m_stats.maxDataBytes = m_storage.getMaxDataSize();

  m_operationsSize += logMemoryUsage(m_requests.getOperationsSize(), "operations", "stream requests");
  m_operationsSize += logMemoryUsage(m_resident.getOperationsSize(), "operations", "stream resident");
  m_operationsSize += logMemoryUsage(m_updates.getOperationsSize(), "operations", "stream updates");
  m_operationsSize += logMemoryUsage(m_storage.getOperationsSize(), "operations", "stream storage");

  res.createBuffer(m_shaderBuffer, sizeof(shaderio::SceneStreaming),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                       | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  NVVK_DBG_NAME(m_shaderBuffer.buffer);

  m_operationsSize += logMemoryUsage(m_shaderBuffer.bufferSize, "operations", "stream shaderio");

  // seed lo res geometry
  initGeometries(res, scene);

  return true;
}

void SceneStreaming::updateBindings(const nvvk::Buffer& sceneBuildingBuffer)
{
  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), m_resources->m_commonBuffers.frameConstants);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), m_resources->m_commonBuffers.readBack);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_GEOMETRIES_SSBO), m_shaderGeometriesBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), sceneBuildingBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), sceneBuildingBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_SSBO), m_shaderBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_UBO), m_shaderBuffer);
  vkUpdateDescriptorSets(m_resources->m_device, writeSets.size(), writeSets.data(), 0, nullptr);
}

void SceneStreaming::resetCachedBlas(Resources::BatchedUploader& uploader)
{
  for(size_t geometryIndex = 0; geometryIndex < m_scene->getActiveGeometryCount(); geometryIndex++)
  {
    SceneStreaming::PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];

    persistentGeometry.cachedBlasUpdateFrame = 0;
    persistentGeometry.cachedBlasLevel       = TRAVERSAL_INVALID_LOD_LEVEL;
    if(persistentGeometry.cachedBlasAllocation)
    {
      m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
    }
  }

  // resets cachedBlasLevel and cachedBlasAddress
  uploader.uploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());
}

void SceneStreaming::resetCachedBlas()
{
  if(m_requiresClas && m_config.allowBlasCaching)
  {
    Resources::BatchedUploader uploader(*m_resources);

    resetCachedBlas(uploader);
    uploader.flush();
  }
}

void SceneStreaming::resetGeometryGroupAddresses(Resources::BatchedUploader& uploader)
{
  // this function fills the geometry group addresses to be invalid
  // except for the persistent lowest detail group

  for(size_t geometryIndex = 0; geometryIndex < m_scene->getActiveGeometryCount(); geometryIndex++)
  {
    SceneStreaming::PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];
    shaderio::Geometry&                 shaderGeometry     = m_shaderGeometries[geometryIndex];
    const Scene::GeometryView&          sceneGeometry      = m_scene->getActiveGeometry(geometryIndex);

    shaderio::LodLevel lastLodLevel = sceneGeometry.lodLevels.back();

    uint64_t* groupAddresses = uploader.uploadBuffer(persistentGeometry.groupAddresses, (uint64_t*)nullptr);
    for(uint32_t groupIndex = 0; groupIndex < lastLodLevel.groupOffset; groupIndex++)
    {
      groupAddresses[groupIndex] = STREAMING_INVALID_ADDRESS_START;
    }
    // except last group, which is always loaded
    groupAddresses[lastLodLevel.groupOffset] = persistentGeometry.lowDetailGroupsData.address;

    // also reset the number of groups loaded per lod-level, except last which is also always loaded.
    uint32_t maxLodLevel = persistentGeometry.lodLevelsCount - 1;
    for(uint32_t i = 0; i < maxLodLevel; i++)
    {
      persistentGeometry.lodLoadedGroupsCount[i] = 0;
    }
    persistentGeometry.lodLoadedGroupsCount[maxLodLevel] = 1;
  }
}

void SceneStreaming::initGeometries(Resources& res, const Scene* scene)
{
  // This function uploads all persistent per-geometry data.
  // - hierarchy nodes for lod traversal
  // - lowest detail geometry group & clusters
  // - the address lookup array to find resident groups
  // It also fills the geometry descriptor stored in
  // m_shaderGeometries

  Resources::BatchedUploader uploader(res);

  m_shaderGeometries.resize(scene->getActiveGeometryCount());
  m_persistentGeometries.resize(scene->getActiveGeometryCount());

  uint32_t instancesOffset = 0;
  for(size_t geometryIndex = 0; geometryIndex < scene->getActiveGeometryCount(); geometryIndex++)
  {
    shaderio::Geometry&                 shaderGeometry     = m_shaderGeometries[geometryIndex];
    SceneStreaming::PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];
    const Scene::GeometryView&          sceneGeometry      = m_scene->getActiveGeometry(geometryIndex);

    size_t numGroups = sceneGeometry.groupInfos.size();
    res.createBufferTyped(persistentGeometry.groupAddresses, numGroups, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(persistentGeometry.groupAddresses.buffer);

    size_t numNodes = sceneGeometry.lodNodes.size();
    res.createBufferTyped(persistentGeometry.nodes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    res.createBufferTyped(persistentGeometry.nodeBboxes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(persistentGeometry.nodes.buffer);
    NVVK_DBG_NAME(persistentGeometry.nodeBboxes.buffer);

    uint32_t numLodLevels = sceneGeometry.lodLevelsCount;
    res.createBufferTyped(persistentGeometry.lodLevels, numLodLevels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(persistentGeometry.lodLevels.buffer);

    m_persistentGeometrySize += persistentGeometry.groupAddresses.bufferSize;
    m_persistentGeometrySize += persistentGeometry.nodes.bufferSize;
    m_persistentGeometrySize += persistentGeometry.nodeBboxes.bufferSize;

    // setup shaderio
    shaderGeometry                         = {};
    shaderGeometry.bbox                    = sceneGeometry.bbox;
    shaderGeometry.nodes                   = persistentGeometry.nodes.address;
    shaderGeometry.nodeBboxes              = persistentGeometry.nodeBboxes.address;
    shaderGeometry.streamingGroupAddresses = persistentGeometry.groupAddresses.address;
    shaderGeometry.lodLevelsCount          = numLodLevels;
    shaderGeometry.lodLevels               = persistentGeometry.lodLevels.address;
    shaderGeometry.cachedBlasAddress       = 0;
    shaderGeometry.cachedBlasLodLevel      = TRAVERSAL_INVALID_LOD_LEVEL;
    shaderGeometry.instancesCount          = sceneGeometry.instanceReferenceCount * scene->getGeometryInstanceFactor();
    shaderGeometry.instancesOffset         = instancesOffset;

    instancesOffset += shaderGeometry.instancesCount;

    persistentGeometry.lodLevelsCount = numLodLevels;
    for(uint32_t i = 0; i < numLodLevels; i++)
    {
      persistentGeometry.lodGroupsCount[i] = sceneGeometry.lodLevels[i].groupCount;
    }

    // basic uploads

    uploader.uploadBuffer(persistentGeometry.nodes, sceneGeometry.lodNodes.data());
    uploader.uploadBuffer(persistentGeometry.nodeBboxes, sceneGeometry.lodNodeBboxes.data());
    uploader.uploadBuffer(persistentGeometry.lodLevels, sceneGeometry.lodLevels.data());

    // seed lowest detail group, which must have just a single cluster
    shaderio::LodLevel     lastLodLevel = sceneGeometry.lodLevels.back();
    const Scene::GroupInfo groupInfo    = sceneGeometry.groupInfos[lastLodLevel.groupOffset];
    Scene::GroupView       groupView(sceneGeometry.groupData, groupInfo);
    assert(groupInfo.clusterCount == 1);

    GeometryGroup geometryGroup     = {uint32_t(geometryIndex), lastLodLevel.groupOffset};
    uint32_t      lastClustersCount = groupInfo.clusterCount;
    uint64_t      lastGroupSize     = groupInfo.getDeviceSize();

    res.createBuffer(persistentGeometry.lowDetailGroupsData, lastGroupSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(persistentGeometry.lowDetailGroupsData.buffer);
    m_persistentGeometrySize += persistentGeometry.lowDetailGroupsData.bufferSize;

    assert(lastClustersCount <= 0xFFFFFFFF);
    assert(m_resident.canAllocateGroup(uint32_t(lastClustersCount)));

    StreamingResident::Group* rgroup = m_resident.addGroup(geometryGroup, lastClustersCount);
    rgroup->deviceAddress            = persistentGeometry.lowDetailGroupsData.address;
    rgroup->lodLevel                 = groupInfo.lodLevel;

    persistentGeometry.lodLoadedGroupsCount[groupInfo.lodLevel] = 1;

    // setup and upload geometry data for the lowest detail group
    void* loGroupData = uploader.uploadBuffer(persistentGeometry.lowDetailGroupsData, (void*)nullptr);

    Scene::fillGroupRuntimeData(groupInfo, groupView, geometryGroup.groupID, rgroup->groupResidentID,
                                rgroup->clusterResidentID, loGroupData, persistentGeometry.lowDetailGroupsData.bufferSize);

    shaderGeometry.lowDetailClusterID = rgroup->clusterResidentID;
    shaderGeometry.lowDetailTriangles = groupInfo.triangleCount;
  }

  // this will set all addresses to invalid, except lowest detail geometry group, which is persistently loaded.
  resetGeometryGroupAddresses(uploader);

  res.createBufferTyped(m_shaderGeometriesBuffer, scene->getActiveGeometryCount(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NVVK_DBG_NAME(m_shaderGeometriesBuffer.buffer);
  m_operationsSize += logMemoryUsage(m_shaderGeometriesBuffer.bufferSize, "operations", "stream geo buffer");

  uploader.uploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  // initial residency table
  m_resident.uploadInitialState(uploader, m_shaderData.resident);

  uploader.flush();
}

void SceneStreaming::cmdBeginFrame(VkCommandBuffer         cmd,
                                   QueueState&             cmdQueueState,
                                   QueueState&             asyncQueueState,
                                   const FrameSettings&    settings,
                                   nvvk::ProfilerGpuTimer& profiler)
{
  // This function sets up all relevant streaming tasks for the frame
  // and configures the content of `m_shaderData` which is uploaded
  // and all streaming related kernels will operate with.
  //
  // The data within `m_shaderData` is stateful and new operations may
  // modify it permanently so that future frames keep the state from the
  // last run operations.
  //
  // - handle completed updates: to give back memory unloads within that update
  // - handle completed storage transfers: to trigger scene updates now that geometry data is available
  // - handle completed request: to trigger new loads/unloads etc.
  //   as a request produces one new update & storage task, these tasks must be handled before
  // - make a new request
  //   likewise a new request requires an empty slot, hence requests must be handled before
  //
  // This function is called by the renderer.

  if(m_frameIndex == 3)
  {
    bool b = true;
  }

  auto     timerSection = profiler.cmdFrameSection(cmd, "Stream Begin");
  VkDevice device       = m_resources->m_device;

  // For each task queue we must ensure that we have one new task index
  // available to acquire for any potential new work in this frame.
  // The ordering in which we drain them matters, as was described above.
  const bool ensureAcquisition = true;

  // pop all completed old updates to recycle as much memory as we can
  while(m_updatesTaskQueue.canPop(device, ensureAcquisition))
  {
    // handleCompletedUpdate
    //
    // The update operation has been completed on the GPU time line, therefore
    // it is safe to fully recycle the memory as it can no longer be reached.
    uint32_t popUpdateIndex = m_updatesTaskQueue.pop();

    const StreamingUpdates::TaskInfo& update = m_updates.getCompletedTask(popUpdateIndex);
    for(uint32_t g = 0; g < update.unloadCount; g++)
    {
      m_storage.free(update.unloadHandles[g]);
    }

    m_updatesTaskQueue.releaseTaskIndex(popUpdateIndex);
  }

  // Our task system allows that new updates can be either
  // handled immediately in the current frame, or decoupled
  // in a later frame.
  //
  // Decoupled allows for asynchronous uploads that can
  // span multiple frames, while immediate means
  // we guarantee transfers completed prior triggering
  // operations.
  uint32_t pushUpdateIndex = INVALID_TASK_INDEX;

  // pop one completed storage transfer
  if(m_storageTaskQueue.canPop(device, ensureAcquisition))
  {
    // handleCompletedStorage
    //
    // The upload of new data was completed, recycle the task and transfer space for future use.
    // If we run in decoupled mode, then push the dependent updates with this frame.
    uint32_t dependentIndex  = INVALID_TASK_INDEX;
    uint32_t popStorageIndex = m_storageTaskQueue.popWithDependent(dependentIndex);
    m_storageTaskQueue.releaseTaskIndex(popStorageIndex);

    // check if we use a decoupled update
    if(dependentIndex != INVALID_TASK_INDEX)
    {
      pushUpdateIndex = dependentIndex;
    }
  }

  bool isImmediateUpdate = false;

  // pop and process one completed request:
  // We read the requested load/unload operations from a completed frame.
  // Within the function we try to make new geometry groups residents,
  // and unloaded ones non-resident.
  // This triggers a storage transfer within the provided command buffer.
  if(m_requestsTaskQueue.canPop(device, ensureAcquisition))
  {
    uint32_t popRequestIndex = m_requestsTaskQueue.pop();

#if 1
    // variant where we pop to latest request
    // otherwise we do process requests in strict order
    while(m_requestsTaskQueue.canPop(device, false))
    {
      // ignore previous request
      m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
      // and use next instead
      popRequestIndex = m_requestsTaskQueue.pop();
    }
#endif

    uint32_t dependentIndex = handleCompletedRequest(cmd, cmdQueueState, asyncQueueState, settings, popRequestIndex);
    // check if immediate update to perform
    if(dependentIndex != INVALID_TASK_INDEX)
    {
      // cannot have deferred and immediate update
      assert(pushUpdateIndex == INVALID_TASK_INDEX);

      pushUpdateIndex   = dependentIndex;
      isImmediateUpdate = true;
    }
  }

  // test if there is an update to be done this frame
  if(pushUpdateIndex != INVALID_TASK_INDEX)
  {
    // Given we know all data was uploaded, we can run the updates to the scene
    // in this frame, which ultimately fulfills a past request on the device.
    //
    // Within this frame compute shaders and other operations, will handle the data
    // provided via values and pointers that are written into.
    // m_shaderData
    //
    // This will mean the current frame can use the new data.

    // both resident and update operations are a synchronized pair, hence
    // single index is sufficient.

    m_resident.applyTask(m_shaderData.resident, pushUpdateIndex, m_frameIndex);
    m_updates.applyTask(m_shaderData.update, pushUpdateIndex, m_frameIndex);

    // we later want to detect the completion of the update task
    // (this was the first thing we did in this function),
    // so push it to task queue
    m_updatesTaskQueue.push(pushUpdateIndex, cmdQueueState.getCurrentState());

    m_lastUpdateIndex = pushUpdateIndex;
  }
  else
  {
    // no patch work this frame
    m_shaderData.update.patchGroupsCount         = 0;
    m_shaderData.update.patchUnloadGroupsCount   = 0;
    m_shaderData.update.patchCachedBlasCount     = 0;
    m_shaderData.update.patchCachedClustersCount = 0;
    m_shaderData.update.loadActiveGroupsOffset   = 0;
    m_shaderData.update.loadActiveClustersOffset = 0;
    m_shaderData.update.newClasCount             = 0;
    m_shaderData.update.taskIndex                = INVALID_TASK_INDEX;
    m_shaderData.update.frameIndex               = m_frameIndex;
  }

  // push new request
  {
    // every frame we will setup new space for new requests made by the device.
    // This is the type of request that we reacted on a few lines above in the
    // `handleCompletedRequest` function.

    uint32_t pushRequestIndex = m_requestsTaskQueue.acquireTaskIndex();
    // the acquisition must be guaranteed by design, as we always handle requests.
    assert(pushRequestIndex != INVALID_TASK_INDEX);

    // get space for request storage
    // and setup this frame's m_shaderData, so that the streaming
    // logic can write to the appropriate pointers.
    m_requests.applyTask(m_shaderData.request, pushRequestIndex, m_frameIndex);
  }

  if(m_requiresClas && m_config.usePersistentClasAllocator)
  {
    // clears the size ranges to zero
    m_clasAllocator.cmdBeginFrame(cmd);
  }

  m_shaderData.frameIndex               = m_frameIndex;
  m_shaderData.ageThreshold             = settings.ageThreshold;
  m_shaderData.useBlasCaching           = settings.useBlasCaching ? 1 : 0;
  m_shaderData.clasPositionTruncateBits = m_clasTriangleInput.minPositionTruncateBitCount;

  // upload final configurations for this frame
  vkCmdUpdateBuffer(cmd, m_shaderBuffer.buffer, 0, sizeof(m_shaderData), &m_shaderData);
}

uint32_t SceneStreaming::handleCompletedRequest(VkCommandBuffer      cmd,
                                                QueueState&          cmdQueueState,
                                                QueueState&          asyncQueueState,
                                                const FrameSettings& settings,
                                                uint32_t             popRequestIndex)
{
  // This function handles the requests from the device to upload new geometry groups,
  // or unload some that haven't been used in a while.
  // The readback of the data is guaranteed to have completed at this point.
  // Uploading will try to handle as much requests as we have memory for.
  // Uploading can be done through an async transfer or on the provided command buffer.
  // After an upload is completed an update task must be run, we can
  // run this task immediately or deferred (see later).
  //
  // Only called in the `SceneStreaming::cmdBeginFrame` function.

  const StreamingRequests::TaskInfo& request = m_requests.getCompletedTask(popRequestIndex);

  // during recording of requests the counters may exceed the limits
  // however the data is always ensured to be within.
  uint32_t loadCount   = std::min(request.shaderData->maxLoads, request.shaderData->loadCounter);
  uint32_t unloadCount = std::min(request.shaderData->maxUnloads, request.shaderData->unloadCounter);

  assert(request.shaderData->errorUpdate == 0 && request.shaderData->errorAgeFilter == 0 && request.shaderData->errorClasNotFound == 0
         && request.shaderData->errorClasAlloc == 0 && request.shaderData->errorClasList == 0
         && request.shaderData->errorClasDealloc == 0 && request.shaderData->errorClasUsedVsAlloc == 0);

  if(m_requiresClas)
  {
    if(m_config.usePersistentClasAllocator)
    {
      m_stats.usedClasBytes   = request.shaderData->clasAllocatedUsedSize;
      m_stats.wastedClasBytes = request.shaderData->clasAllocatedWastedSize;
      m_stats.maxSizedLeft    = request.shaderData->clasAllocatedMaxSizedLeft;
    }
    else
    {
      m_stats.usedClasBytes   = request.shaderData->clasCompactionUsedSize;
      m_stats.wastedClasBytes = 0;
      m_stats.maxSizedLeft = uint32_t((m_config.maxClasMegaBytes * 1024 * 1024 - request.shaderData->clasCompactionUsedSize)
                                      / (m_clasSingleMaxSize * m_scene->m_config.clusterGroupSize));
    }
  }

#if !STREAMING_DEBUG_FORCE_REQUESTS
  if((!loadCount && !unloadCount) || !m_debugFrameLimit)
  {
    // no work to do
    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
    return INVALID_TASK_INDEX;
  }
#endif

  // for debugging
  if(m_debugFrameLimit > 0)
    m_debugFrameLimit--;

  uint32_t pushStorageIndex = m_storageTaskQueue.acquireTaskIndex();
  uint32_t pushUpdateIndex  = m_updatesTaskQueue.acquireTaskIndex();

  // early out if we are not able to acquire both tasks to serve the request
  if(pushStorageIndex == INVALID_TASK_INDEX || pushUpdateIndex == INVALID_TASK_INDEX)
  {
    // give back acquisitions we don't make use of
    if(pushStorageIndex != INVALID_TASK_INDEX)
    {
      m_storageTaskQueue.releaseTaskIndex(pushStorageIndex);
    }
    if(pushUpdateIndex != INVALID_TASK_INDEX)
    {
      m_updatesTaskQueue.releaseTaskIndex(pushUpdateIndex);
    }
    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
    return INVALID_TASK_INDEX;
  }

  StreamingStorage::TaskInfo& storageTask = m_storage.getNewTask(pushStorageIndex);
  StreamingUpdates::TaskInfo& updateTask  = m_updates.getNewTask(pushUpdateIndex);

  bool useBlasCaching = m_requiresClas && m_config.allowBlasCaching && settings.useBlasCaching;

  // let's do unloads first, so we can recycle resident objects
  for(uint32_t g = 0; g < unloadCount; g++)
  {
    GeometryGroup geometryGroup = request.unloadGeometryGroups[g];

    assert(geometryGroup.geometryID < m_scene->getActiveGeometryCount());
    assert(geometryGroup.groupID < m_scene->getActiveGeometry(geometryGroup.geometryID).totalClustersCount);

    const StreamingResident::Group* group = m_resident.findGroup(geometryGroup);
    if(!group)
    {
      // The group might already be removed through a previous request.
      // This can happen cause it may take a while until the patch that really removes something
      // is applied on GPU timeline.
      continue;
    }

    // setup patch
    uint32_t                  unloadIndex = updateTask.unloadCount++;
    shaderio::StreamingPatch& patch       = updateTask.unloadPatches[unloadIndex];
    patch.geometryID                      = geometryGroup.geometryID;
    patch.groupID                         = geometryGroup.groupID;
    patch.groupAddress                    = STREAMING_INVALID_ADDRESS_START;

    // note actual storage memory cannot be recycled here, cause only
    // once the new "update" operation was completed, the gpu's scene graph
    // will not use the data anymore.
    // So defer the actual unloading to the `SceneStreaming::handleCompletedUpdate`
    // above.
    assert(group->storageHandle);
    updateTask.unloadHandles[unloadIndex] = group->storageHandle;

    assert(m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel] > 0);
    m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel]--;

    // and remove from active resident
    m_resident.removeGroup(group->groupResidentID);

    // append to geometry patch list if necessary
    if(useBlasCaching && m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame != m_frameIndex)
    {
      m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame = m_frameIndex;
      uint32_t                          geometryPatchIndex                   = updateTask.geometryCachedCount++;
      shaderio::StreamingGeometryPatch& geometryPatch = updateTask.geometryPatches[geometryPatchIndex];
      geometryPatch.geometryID                        = geometryGroup.geometryID;
    }
  }

  // for ray tracing
  // we have two different clas memory management systems and for both
  // one needs to see how much space is left for future allocations
  // - move clas to be compacted all the time
  uint64_t clasMovedUsedSize     = request.shaderData->clasCompactionUsedSize;
  uint64_t clasMovedReservedSize = m_config.maxClasMegaBytes * 1024 * 1024;
  // - a persistent allocator implemented on the gpu.
  uint32_t clasAllocatedMaxSizedLeft = request.shaderData->clasAllocatedMaxSizedLeft;

  // Need to account for clas operations that happen on the gpu timeline after this request's
  // frame. They indirectly reduce the budget we are guaranteed to have left for building new clas.
  StreamingUpdates::NewInfo futureNew = m_updates.getFutureNew(request.shaderData->frameIndex);
  clasMovedUsedSize += m_clasSingleMaxSize * futureNew.clusters;
  clasAllocatedMaxSizedLeft -= std::min(clasAllocatedMaxSizedLeft, futureNew.groups);

  uint32_t clasBuildOffset = 0;
  uint64_t clasBuildSize   = 0;

  // all newly added groups will be appended to the active list
  updateTask.loadActiveGroupsOffset   = m_resident.getLoadActiveGroupsOffset();
  updateTask.loadActiveClustersOffset = m_resident.getLoadActiveClustersOffset();

  uint64_t transferBytes = 0;

  m_stats.couldNotAllocateClas  = 0;
  m_stats.couldNotTransfer      = 0;
  m_stats.couldNotAllocateGroup = 0;
  m_stats.couldNotStore         = 0;
  m_stats.uncompletedLoadCount  = 0;

  for(uint32_t g = 0; g < loadCount; g++)
  {
    GeometryGroup geometryGroup = request.loadGeometryGroups[g];

    assert(geometryGroup.geometryID < m_scene->getActiveGeometryCount());
    assert(geometryGroup.groupID < m_scene->getActiveGeometry(geometryGroup.geometryID).totalClustersCount);

    if(m_resident.findGroup(geometryGroup))
    {
      // It could take more than one frame until the patch that handles the load
      // is activated on the GPU timeline, and until then the same requests might be
      // made.

      continue;
    }

    const Scene::GeometryView& sceneGeometry = m_scene->getActiveGeometry(geometryGroup.geometryID);

    // figure out size of this geometry group.
    // This includes all relevant cluster data, including vertices, triangle indices...
    const Scene::GroupInfo groupInfo       = sceneGeometry.groupInfos[geometryGroup.groupID];
    uint32_t               clusterCount    = groupInfo.clusterCount;
    uint64_t               groupDeviceSize = groupInfo.getDeviceSize();
    uint64_t               groupClasSize   = 0;
    bool                   canAllocateClas = true;

    if(m_requiresClas)
    {
      groupClasSize = m_clasSingleMaxSize * clusterCount;

      // must always fit in scratch
      assert((clasBuildSize + groupClasSize) <= m_clasScratchNewClasSize);

      if(m_config.usePersistentClasAllocator)
      {
        canAllocateClas = clasAllocatedMaxSizedLeft > 0;
      }
      else
      {
        canAllocateClas = (clasMovedUsedSize + (clasBuildSize + groupClasSize)) <= clasMovedReservedSize;
      }
    }

    uint64_t                  deviceAddress;
    nvvk::BufferSubAllocation storageHandle;

    bool canTransfer      = m_storage.canTransfer(storageTask, groupDeviceSize);
    bool canStore         = m_storage.allocate(storageHandle, geometryGroup, groupDeviceSize, deviceAddress);
    bool canAllocateGroup = m_resident.canAllocateGroup(clusterCount);

    // test if we can allocate
    if(!canTransfer || !canStore || !canAllocateGroup || !canAllocateClas)
    {
      m_stats.couldNotAllocateClas += (!canAllocateClas);
      m_stats.couldNotTransfer += (!canTransfer);
      m_stats.couldNotAllocateGroup += (!canAllocateGroup);
      m_stats.couldNotStore += (!canStore);

      if(canStore)
      {
        // return memory on failure
        m_storage.free(storageHandle);
      }

      if(clusterCount < 8)
      {
        m_stats.uncompletedLoadCount += loadCount - g;
        break;  // heuristic if small groups don't fit anymore then we fully break
      }
      else
      {
        m_stats.uncompletedLoadCount++;
        continue;
      }
    }

    StreamingResident::Group* residentGroup = m_resident.addGroup(geometryGroup, clusterCount);
    residentGroup->storageHandle            = storageHandle;
    residentGroup->deviceAddress            = deviceAddress;
    residentGroup->lodLevel                 = groupInfo.lodLevel;
    void* groupData                         = m_storage.appendTransfer(storageTask, residentGroup->storageHandle);

    assert(deviceAddress % 16 == 0);

    {
      Scene::GroupView groupView(sceneGeometry.groupData, groupInfo);
      if(groupInfo.uncompressedSizeBytes)
      {
        Scene::decompressGroup(groupInfo, groupView, groupData, groupDeviceSize);
      }
      else
      {
        // simply copy data as is, the streaming patch will take care of modifying the data
        // where needed
        memcpy(groupData, groupView.raw, groupView.rawSize);
      }
    }

    m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[groupInfo.lodLevel]++;

    // append to geometry patch list if necessary
    if(useBlasCaching && m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame != m_frameIndex)
    {
      m_persistentGeometries[geometryGroup.geometryID].cachedBlasUpdateFrame = m_frameIndex;
      uint32_t                          geometryPatchIndex                   = updateTask.geometryCachedCount++;
      shaderio::StreamingGeometryPatch& geometryPatch = updateTask.geometryPatches[geometryPatchIndex];
      geometryPatch.geometryID                        = geometryGroup.geometryID;
    }

    // setup patch
    shaderio::StreamingPatch& patch = updateTask.loadPatches[updateTask.loadCount++];
    patch.geometryID                = geometryGroup.geometryID;
    patch.groupID                   = geometryGroup.groupID;
    patch.groupAddress              = deviceAddress;
    patch.groupResidentID           = residentGroup->groupResidentID;
    patch.clusterResidentID         = residentGroup->clusterResidentID;
    patch.clasBuildOffset           = clasBuildOffset;
    patch.clusterCount              = groupInfo.clusterCount;
    patch.lodLevel                  = groupInfo.lodLevel;

    clasBuildOffset += clusterCount;
    clasBuildSize += groupClasSize;
    clasAllocatedMaxSizedLeft--;

    // stats
    transferBytes += groupInfo.sizeBytes;
  }

  updateTask.newClusterCount = clasBuildOffset;

#if !STREAMING_DEBUG_FORCE_REQUESTS
  if(updateTask.loadCount == 0 && updateTask.unloadCount == 0)
  {
    // we ended up doing no work
    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
    m_updatesTaskQueue.releaseTaskIndex(pushUpdateIndex);
    m_storageTaskQueue.releaseTaskIndex(pushStorageIndex);
    return INVALID_TASK_INDEX;
  }
#endif

  if(m_config.useAsyncTransfer)
  {
    // don't use immediate command buffer from main queue,
    // but use transfer queue instead.

    NVVK_CHECK(m_storage.m_taskCommandPool.acquireCommandBuffer(pushStorageIndex, cmd));
    VkCommandBufferBeginInfo cmdInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(cmd, &cmdInfo);
  }


  // evaluate geometry lod state and see if we need to
  // rebuild the cached blas
  if(useBlasCaching)
  {
    handleBlasCaching(updateTask, settings);
  }

  uint32_t transferCount = 0;
  // finalize data for completed new residency & patch
  // residency & updates always operate in synchronized pairs
  transferBytes += m_updates.cmdUploadTask(cmd, pushUpdateIndex);
  transferBytes += m_resident.cmdUploadTask(cmd, pushUpdateIndex);
  transferCount += m_storage.cmdUploadTask(cmd);
  transferCount += 2;

  if(updateTask.loadCount)
  {
    // only log to stats for loads
    m_stats.transferBytes = transferBytes;
    m_stats.transferCount = transferCount;

    m_stats.loadCount = updateTask.loadCount;
  }
  if(updateTask.unloadCount)
  {
    m_stats.unloadCount = updateTask.unloadCount;
  }

  // When we use async we can either wait until async completed (can take more than a frame)
  // or we guarantee it completes for the frame we are currently preparing within `cmd`.
  // When not using async we always know the transfer completes within the current frame.

  bool useDecoupledUpdate = m_config.useAsyncTransfer && m_config.useDecoupledAsyncTransfer;

  nvvk::SemaphoreState storageSemaphoreState =
      m_config.useAsyncTransfer ? asyncQueueState.getCurrentState() : cmdQueueState.getCurrentState();

  if(m_config.useAsyncTransfer)
  {
    vkEndCommandBuffer(cmd);

    if(!m_config.useDecoupledAsyncTransfer)
    {
      // if not using decoupled, then let immediate command buffer's queue wait for this
      // transfer to be completed

      // get wait from async queue
      VkSemaphoreSubmitInfo semWaitInfo = asyncQueueState.getWaitSubmit(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
      // push it for use in primary queue
      cmdQueueState.m_pendingWaits.push_back(semWaitInfo);
    }

    // trigger async transfer queue submit
    VkSemaphoreSubmitInfo     semSubmitInfo = asyncQueueState.advanceSignalSubmit(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    VkCommandBufferSubmitInfo cmdBufInfo    = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdBufInfo.commandBuffer                = cmd;

    VkSubmitInfo2 submits            = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR};
    submits.pCommandBufferInfos      = &cmdBufInfo;
    submits.commandBufferInfoCount   = 1;
    submits.pSignalSemaphoreInfos    = &semSubmitInfo;
    submits.signalSemaphoreInfoCount = 1;
    vkQueueSubmit2(asyncQueueState.m_queue, 1, &submits, nullptr);
  }

  // give back the index for future write operations
  m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);

  // enqueue the storage task
  // the dependentIndex may be set to the pushUpdateIndex if we use decoupled
  m_storageTaskQueue.push(pushStorageIndex, storageSemaphoreState, useDecoupledUpdate ? pushUpdateIndex : INVALID_TASK_INDEX);

  // otherwise, we return update task to be handled directly in this frame
  return useDecoupledUpdate ? INVALID_TASK_INDEX : pushUpdateIndex;
}

void SceneStreaming::handleBlasCaching(StreamingUpdates::TaskInfo& updateTask, const FrameSettings& settings)
{
  uint32_t writeIndex = 0;

  uint32_t cachedBuildsTotal   = 0;
  uint32_t cachedClustersTotal = 0;

#if STREAMING_DEBUG_FORCE_REQUESTS
  if(updateTask.geometryCachedCount == 0)
  {
    updateTask.geometryPatches[updateTask.geometryCachedCount++].geometryID = 0;
  }
#endif

  for(uint32_t g = 0; g < updateTask.geometryCachedCount; g++)
  {
    shaderio::StreamingGeometryPatch sgpatch            = updateTask.geometryPatches[g];
    PersistentGeometry&              persistentGeometry = m_persistentGeometries[sgpatch.geometryID];
    const Scene::GeometryView&       geometryView       = m_scene->getActiveGeometry(sgpatch.geometryID);

    uint32_t cachedClustersCount = 0;
    uint32_t blasCacheMinLevel =
        persistentGeometry.lodLevelsCount - std::min(settings.blasCacheMinLevel, persistentGeometry.lodLevelsCount);

    // skip last level, always exists as low detail blas
    for(uint32_t i = blasCacheMinLevel; i < persistentGeometry.lodLevelsCount - 1; i++)
    {
      // fully loaded
      if(persistentGeometry.lodGroupsCount[i] == persistentGeometry.lodLoadedGroupsCount[i])
      {
        uint32_t groupCount          = geometryView.lodLevels[i].groupCount;
        uint32_t groupOffset         = geometryView.lodLevels[i].groupOffset;
        uint32_t cachedClustersCount = geometryView.lodLevels[i].clusterCount;

        // check if it fits
        if(cachedClustersCount <= STREAMING_CACHED_BLAS_MAX_CLUSTERS)
        {
          sgpatch.cachedBlasLodLevel = i;
          break;
        }
        else
        {
          cachedClustersCount = 0;
        }
      }
    }

    // three scenarios:
    // lower detail resident than before: must rebuild cached blas or invalidate
    // higher detail resident than before: can rebuild cached blas otherwise use existing
    // do nothing

    bool isInvalidateOnly = !cachedClustersCount && persistentGeometry.cachedBlasLevel != TRAVERSAL_INVALID_LOD_LEVEL;
    bool isLowerDetail    = cachedClustersCount && sgpatch.cachedBlasLodLevel > persistentGeometry.cachedBlasLevel;
    bool isHigherDetail   = cachedClustersCount && sgpatch.cachedBlasLodLevel < persistentGeometry.cachedBlasLevel
                          && persistentGeometry.cachedBlasUpdateFrame + settings.blasCacheAgeThreshold > m_frameIndex;

    if(isLowerDetail || isHigherDetail || isInvalidateOnly || STREAMING_DEBUG_FORCE_REQUESTS)
    {
      if(isLowerDetail || isInvalidateOnly)
      {
        // de-allocate first, because will use less space next, which is guaranteed to fit
        if(persistentGeometry.cachedBlasAllocation)
        {
          m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
        }
      }

      bool canBuild = !isInvalidateOnly && (cachedClustersTotal + cachedClustersCount <= settings.blasCacheMaxClusters)
                      && (cachedBuildsTotal + 1 <= settings.blasCacheMaxBuilds);

      // attempt to allocate
      nvvk::BufferSubAllocation subAllocation;
      canBuild = canBuild && allocateCachedBlas(persistentGeometry, cachedClustersCount, settings, subAllocation);

      if(canBuild)
      {
        // able to allocate & build new

        // de-allocate old if still exists
        // isHigherDetail attempts to get space for higher detail first,
        // so it can keep old cached blas if building fails
        if(persistentGeometry.cachedBlasAllocation)
        {
          m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
        }
        persistentGeometry.cachedBlasLevel       = sgpatch.cachedBlasLodLevel;
        persistentGeometry.cachedBlasAllocation  = subAllocation;
        persistentGeometry.cachedBlasUpdateFrame = m_frameIndex;

        // setup cached build patch
        sgpatch.cachedBlasAddress       = m_cachedBlasAllocator.subRange(subAllocation).address;
        sgpatch.cachedBlasClustersCount = uint16_t(cachedClustersCount);

        updateTask.geometryPatches[writeIndex++] = sgpatch;

        cachedClustersTotal += cachedClustersCount;
        cachedBuildsTotal++;
      }
      else if(isHigherDetail)
      {
        // we were not able to change to higher detail, leave things as is
      }
      else
      {
        // setup invalidate patch
        persistentGeometry.cachedBlasLevel       = TRAVERSAL_INVALID_LOD_LEVEL;
        persistentGeometry.cachedBlasUpdateFrame = m_frameIndex;

        sgpatch.cachedBlasLodLevel = TRAVERSAL_INVALID_LOD_LEVEL;
        sgpatch.cachedBlasAddress  = 0;

        updateTask.geometryPatches[writeIndex++] = sgpatch;
      }
    }
  }

  updateTask.geometryCachedCount         = writeIndex;
  updateTask.geometryCachedClustersCount = cachedClustersTotal;
}

bool SceneStreaming::allocateCachedBlas(const PersistentGeometry&  geometry,
                                        uint32_t                   lodClustersCount,
                                        const FrameSettings&       settings,
                                        nvvk::BufferSubAllocation& subAllocation)
{
  VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
  // Just using m_hiPerGeometryClusters here is problematic, as the intermediate state
  // of a continuous lod can yield higher numbers (especially when streaming may temporarily cause overlapping of different levels).
  // Therefore, we use the highest sum of all clusters across all lod levels.
  blasInput.maxClusterCountPerAccelerationStructure = lodClustersCount;
  blasInput.maxTotalClusterCount                    = lodClustersCount;

  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = 1;
  inputs.opMode                                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputs.opInput.pClustersBottomLevel = &blasInput;
  inputs.flags                        = settings.blasCacheFlags;

  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(m_resources->m_device, &inputs, &sizesInfo);

  return m_cachedBlasAllocator.subAllocate(subAllocation, sizesInfo.accelerationStructureSize, m_cachedBlasAlignment) == VK_SUCCESS;
}

static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  // compute workgroup count from threads
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void SceneStreaming::cmdPreTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, nvvk::ProfilerGpuTimer& profiler)
{
  Resources& res = *m_resources;
  // Prior traversal we run the update task.
  // This modifies the device address array of geometry groups so that
  // traversal knows whether a geometry group is resident or not and where to
  // find it.
  // It also handles the unloading by invalidating such addresses.
  //
  // For ray tracing we are building new clusters for newly loaded
  // cluster groups.
  //
  // This function is called by the renderer.

  auto timerSection = profiler.cmdFrameSection(cmd, "Stream Pre Traversal");

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);


  if(m_requiresClas && m_config.usePersistentClasAllocator)
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Clas Deallocate Groups");
    if(m_shaderData.update.patchUnloadGroupsCount)
    {
      // this dispatch will handle giving back clas memory of unloaded groups
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAllocatorUnloadGroups);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.update.patchUnloadGroupsCount, STREAM_ALLOCATOR_UNLOAD_GROUPS_WORKGROUP));

      // must not overlap with the next dispatch that actually removes groups, as our pointers would be invalid
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }
  }

  // if we have an update to perform do it prior traversal
  if(m_shaderData.update.patchGroupsCount || m_shaderData.update.patchCachedBlasCount)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_requiresClas ? m_pipelines.computeUpdateSceneRay : m_pipelines.computeUpdateSceneRaster);

    res.cmdLinearDispatch(cmd, getWorkGroupCount(std::max(m_shaderData.update.patchGroupsCount, m_shaderData.update.patchCachedBlasCount),
                                                 STREAM_UPDATE_SCENE_WORKGROUP));

    // with the update also comes a new compacted list of resident objects
    m_resident.cmdRunTask(cmd, m_shaderData.update.taskIndex);
  }

  // rasterization ends here
  if(!m_requiresClas)
    return;

  // wait for previous update & transfer to complete

  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
                       1, &memBarrier, 0, nullptr, 0, nullptr);

  // When ray tracing we require to build new CLAS for all newly loaded groups.
  // We build them into scratch space, and then later (`cmdPostTraversal`) move
  // them to their resident locations.

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Clas Build New");

    uint32_t newClasCount = m_shaderData.update.newClasCount;

    VkClusterAccelerationStructureTriangleClusterInputNV clasTriangleInput = m_clasTriangleInput;
    // adjust for actual task
    clasTriangleInput.maxTotalTriangleCount = m_clasTriangleInput.maxClusterTriangleCount * newClasCount;
    clasTriangleInput.maxTotalVertexCount   = m_clasTriangleInput.maxClusterVertexCount * newClasCount;

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.input = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    cmdInfo.input.flags                         = m_config.clasBuildFlags;
    cmdInfo.input.maxAccelerationStructureCount = newClasCount;
    cmdInfo.input.opInput.pTriangleClusters     = &clasTriangleInput;
    cmdInfo.input.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    cmdInfo.input.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;

    cmdInfo.srcInfosArray.deviceAddress = m_shaderData.update.newClasBuilds;
    cmdInfo.srcInfosArray.stride        = sizeof(shaderio::ClasBuildInfo);
    cmdInfo.srcInfosArray.size          = sizeof(shaderio::ClasBuildInfo) * newClasCount;

    cmdInfo.dstAddressesArray.deviceAddress = m_shaderData.update.newClasAddresses;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * newClasCount;

    cmdInfo.dstSizesArray.deviceAddress = m_shaderData.update.newClasSizes;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);
    cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * newClasCount;

    cmdInfo.srcInfosCount = 0;

    cmdInfo.dstImplicitData = clasScratchBuffer;

    cmdInfo.scratchData = clasScratchBuffer + m_clasScratchNewClasSize;
    assert(cmdInfo.scratchData % m_clasScratchAlignment == 0);

    // do condition here to always trigger timers, keeps ui stable
    if(newClasCount)
    {
      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
    }
  }

  if(m_config.usePersistentClasAllocator)
  {
    // When we use the persistent clas memory allocator we need to find
    // empty gaps to allocate the persistent location for our newly built clas from.

    auto timerSection = profiler.cmdFrameSection(cmd, "Clas Prep Allocation");

    if(m_shaderData.update.patchGroupsCount || STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS)
    {
      // there are load or unload operations, so compute the free gaps
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAllocatorBuildFreeGaps);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.clasAllocator.sectorCount,
                                                   STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP / SUBGROUP_SIZE));

      // if we need to allocate new group clusters (loaded new groups) then we need the full detail
      // binned list of free gaps, otherwise we are fine with just knowing the counters
      if((m_shaderData.update.patchGroupsCount > m_shaderData.update.patchUnloadGroupsCount) || STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS)
      {
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        // after computing the free gaps, prepare their insertion in the binned lists

        // this setup computes the launch grid for the indirect dispatch further down, and resets some internal atomics
        uint32_t streamSetup = STREAM_SETUP_ALLOCATOR_FREEINSERT;
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(streamSetup), &streamSetup);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeSetup);
        vkCmdDispatch(cmd, 1, 1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        ;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        // this dispatch handles the offset computation where each size-based free range starts
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAllocatorSetupInsertion);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.clasAllocator.maxAllocationSize,
                                                     STREAM_ALLOCATOR_SETUP_INSERTION_WORKGROUP));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        // the last dispatch handles binning all the free gaps based on their size into the appropriate free list ranges
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAllocatorFreeGapsInsert);
        vkCmdDispatchIndirect(cmd, m_shaderBuffer.buffer,
                              offsetof(shaderio::SceneStreaming, clasAllocator)
                                  + offsetof(shaderio::StreamingAllocator, dispatchFreeGapsInsert));
      }
    }
  }
  else
  {
    // For the clas compaction scheme we compute the move operations for all old clas

    auto timerSection = profiler.cmdFrameSection(cmd, "Clas Offsets Old");

    if(m_shaderData.update.patchUnloadGroupsCount)
    {
      // Only run compaction if we had unloads.
      // Compute compaction of old clusters here, however the actual move is performed in `cmdPostTraversal`
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeCompactionClasOld);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.update.loadActiveGroupsOffset, STREAM_COMPACTION_OLD_CLAS_WORKGROUP));
    }
    else
    {
      // Without any unloads happening, there is no need to compact, and so we just use this setup
      // kernel to update internal state on the gpu-timeline from past frames' compaction state.
      uint32_t streamSetup = STREAM_SETUP_COMPACTION_OLD_NO_UNLOADS;
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(streamSetup), &streamSetup);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeSetup);
      vkCmdDispatch(cmd, 1, 1, 1);
    }
  }
}

void SceneStreaming::cmdPostTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, bool runAgeFilter, nvvk::ProfilerGpuTimer& profiler)
{
  Resources& res = *m_resources;

  // After traversal was performed, this function filters resident cluster groups
  // by age to append to the unload request list.
  // The traversal itself will have appended load requests and reset the age of
  // used cluster groups.
  //
  // For ray tracing we compact all resident clusters and append (also compacted)
  // the newly build clusters from the previous `cmdPreTraversal` step.
  //
  // This function is called by the renderer.

  auto timerSection = profiler.cmdFrameSection(cmd, "Stream Post Traversal");

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

  if(m_shaderData.resident.activeGroupsCount && runAgeFilter)
  {
    // age filter resident groups, writes unload request array

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAgeFilterGroups);
    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.resident.activeGroupsCount, STREAM_AGEFILTER_GROUPS_WORKGROUP));
  }

  // rasterization ends here
  if(!m_requiresClas)
    return;


  // When ray tracing we need to manage the storage of the newly built CLAS

  if(m_config.usePersistentClasAllocator)
  {
    // Let's allocate clas memory for all clas of the newly loaded groups.
    // This also computes the move operations from newly built clas in scratch
    // space to their resident locations.

    auto timerSection = profiler.cmdFrameSection(cmd, "Clas Allocate New");

    uint32_t patchLoadGroupsCount = m_shaderData.update.patchGroupsCount - m_shaderData.update.patchUnloadGroupsCount;

    if(patchLoadGroupsCount)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAllocatorLoadGroups);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(patchLoadGroupsCount, STREAM_ALLOCATOR_LOAD_GROUPS_WORKGROUP));
    }
  }
  else
  {
    // In the compaction based scheme we will move all old clas to the beginning
    // of the memory range, so that newly built can be appended at the end.

    auto timerSection = profiler.cmdFrameSection(cmd, "Clas Compact Old");

    // requires `m_pipelines.computeMoveClasOld` to have been run

    uint32_t oldClasCount = m_shaderData.update.loadActiveClustersOffset;

    VkClusterAccelerationStructureMoveObjectsInputNV clasMoveInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV};
    clasMoveInput.type          = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV;
    clasMoveInput.maxMovedBytes = uint32_t(
        std::min(size_t(uint32_t(~0)), std::min(m_clasSingleMaxSize * oldClasCount, m_config.maxClasMegaBytes * 1024 * 1024)));
    // old clusters can overlap themselves
    clasMoveInput.noMoveOverlap = VK_FALSE;

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.input = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    cmdInfo.input.maxAccelerationStructureCount = oldClasCount;
    cmdInfo.input.opInput.pMoveObjects          = &clasMoveInput;
    cmdInfo.input.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
    cmdInfo.input.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

    cmdInfo.srcInfosArray.deviceAddress = m_shaderData.update.moveClasSrcAddresses;
    cmdInfo.srcInfosArray.stride        = sizeof(uint64_t);
    cmdInfo.srcInfosArray.size          = sizeof(uint64_t) * oldClasCount;

    cmdInfo.dstAddressesArray.deviceAddress = m_shaderData.update.moveClasDstAddresses;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * oldClasCount;

    cmdInfo.srcInfosCount = m_shaderBuffer.address + offsetof(shaderio::SceneStreaming, update)
                            + offsetof(shaderio::StreamingUpdate, moveClasCounter);

    cmdInfo.scratchData = clasScratchBuffer + m_clasScratchNewClasSize;
    assert(cmdInfo.scratchData % m_clasScratchAlignment == 0);

    // do condition here to always trigger timers, keeps ui stable
    if(m_shaderData.update.patchUnloadGroupsCount && oldClasCount && !STREAMING_DEBUG_WITHOUT_RT)
    {
      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
    }

    // in the compaction scheme we need to compute the newly built clas move locations
    // after we moved the old.
    if(m_shaderData.update.newClasCount)
    {
      if(m_shaderData.update.patchUnloadGroupsCount && oldClasCount)
      {
        // wait for completion of previous move, since we overwrite the move argument array next
        VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV;
        memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }

      // compute the move operations for newly built clas
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeCompactionClasNew);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.update.newClasCount, STREAM_COMPACTION_NEW_CLAS_WORKGROUP));
    }
  }

  {
    // Pre-traversal we have built the new clusters into scratch space,
    // now we need to move them into their resident location.
    //
    // This is true for both persistent clas allocator, as well as the simple compaction scheme.

    auto timerSection = profiler.cmdFrameSection(cmd, "Clas Append New");

    // wait for completion
    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    VkClusterAccelerationStructureMoveObjectsInputNV clasMoveInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV};
    clasMoveInput.type          = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV;
    clasMoveInput.maxMovedBytes = m_clasScratchNewClasSize;
    // no overlap as we copy from dedicated scratch build new to final destination
    clasMoveInput.noMoveOverlap = VK_TRUE;

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.input = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    cmdInfo.input.maxAccelerationStructureCount = m_shaderData.update.newClasCount;
    cmdInfo.input.opInput.pMoveObjects          = &clasMoveInput;
    cmdInfo.input.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
    cmdInfo.input.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

    cmdInfo.dstAddressesArray.deviceAddress = m_shaderData.update.moveClasDstAddresses;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_shaderData.update.newClasCount;

    cmdInfo.srcInfosArray.deviceAddress = m_shaderData.update.moveClasSrcAddresses;
    cmdInfo.srcInfosArray.stride        = sizeof(uint64_t);
    cmdInfo.srcInfosArray.size          = sizeof(uint64_t) * m_shaderData.update.newClasCount;

    cmdInfo.srcInfosCount = 0;

    // start scratch after the newly built clas
    cmdInfo.scratchData = clasScratchBuffer + m_clasScratchNewClasSize;
    assert(cmdInfo.scratchData % m_clasScratchAlignment == 0);

    // do condition here to always trigger timers, keeps ui stable
    if(m_shaderData.update.newClasCount && !(STREAMING_DEBUG_WITHOUT_RT || STREAMING_DEBUG_MANUAL_MOVE))
    {
      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
    }
  }

  {
    // always run the status operation, it stores information in the `request` that allows
    // us to know how much clas memory is in use and how many new clas we are guaranteed to
    // be able to load.

    uint32_t specialID = m_config.usePersistentClasAllocator ? STREAM_SETUP_ALLOCATOR_STATUS : STREAM_SETUP_COMPACTION_STATUS;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(specialID), &specialID);
    vkCmdDispatch(cmd, 1, 1, 1);
  }
}

void SceneStreaming::cmdEndFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, nvvk::ProfilerGpuTimer& profiler)
{
  // Perform the request readback.
  // we pass the location of `shaderio::StreamingRequest` within m_streamingBuffer, as it contains
  // the counter values for how much loads/unloads to perform as well as how much memory
  // for ray tracing CLAS is currently in use.
  //
  // This function is called by the renderer.

  auto timerSection = profiler.cmdFrameSection(cmd, "Stream End");

  m_requests.cmdRunTask(cmd, m_shaderData.request, m_shaderBuffer.buffer, offsetof(shaderio::SceneStreaming, request));

  m_requestsTaskQueue.push(m_shaderData.request.taskIndex, cmdQueueState.getCurrentState());

  m_frameIndex++;
}

void SceneStreaming::getStats(StreamingStats& stats) const
{
  stats = m_stats;

  m_storage.getStats(stats);
  m_resident.getStats(stats);
  stats.persistentDataBytes = m_persistentGeometrySize;
  stats.persistentClasBytes = m_clasLowDetailSize;
}

size_t SceneStreaming::getClasSize(bool reserved) const
{
  if(reserved)
  {
    return m_clasLowDetailSize + m_stats.reservedClasBytes;
  }
  else
  {
    return m_clasLowDetailSize + m_stats.usedClasBytes;
  }
}

size_t SceneStreaming::getBlasSize(bool reserved) const
{
  size_t size = m_blasSize;

  if(m_requiresClas && m_config.allowBlasCaching)
  {
    nvvk::BufferSubAllocator::Report report = m_cachedBlasAllocator.getReport();

    if(reserved)
    {
      size += report.reservedSize;
    }
    else
    {
      size += report.requestedSize;
    }
  }

  return size;
}

size_t SceneStreaming::getGeometrySize(bool reserved) const
{
  StreamingStats stats;
  getStats(stats);

  if(reserved)
  {
    return m_persistentGeometrySize + stats.reservedDataBytes;
  }
  else
  {
    return m_persistentGeometrySize + stats.usedDataBytes;
  }
}

bool SceneStreaming::updateClasRequired(bool state)
{
  if(state != m_requiresClas)
  {
    if(state)
    {
      return initClas();
    }
    else
    {
      deinitClas();
    }
  }

  return true;
}

void SceneStreaming::deinit()
{
  if(!m_resources)
    return;

  Resources& res = *m_resources;

  deinitClas();

  deinitShadersAndPipelines();
  m_dsetPack.deinit();
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);

  m_resident.deinit(res);
  m_storage.deinit(res);
  m_updates.deinit(res);
  m_requests.deinit(res);

  for(auto& it : m_persistentGeometries)
  {
    res.m_allocator.destroyBuffer(it.groupAddresses);
    res.m_allocator.destroyBuffer(it.nodeBboxes);
    res.m_allocator.destroyBuffer(it.nodes);
    res.m_allocator.destroyBuffer(it.lodLevels);
    res.m_allocator.destroyBuffer(it.lowDetailGroupsData);
  }
  m_persistentGeometries.clear();

  res.m_allocator.destroyBuffer(m_shaderGeometriesBuffer);
  res.m_allocator.destroyBuffer(m_shaderBuffer);

  m_resources = nullptr;
  m_scene     = nullptr;
}

void SceneStreaming::reset()
{
  Resources& res = *m_resources;

  vkDeviceWaitIdle(res.m_device);

  m_debugFrameLimit = s_defaultDebugFrameLimit;

  m_requestsTaskQueue = {};
  m_storageTaskQueue  = {};
  m_updatesTaskQueue  = {};

  // reset resident objects to just roots
  m_resident.reset(m_shaderData.resident);
  m_updates.reset();

  // reset dynamic storage
  m_storage.reset();

  // need to reset internal clock
  m_frameIndex = 1;

  Resources::BatchedUploader uploader(res);
  resetGeometryGroupAddresses(uploader);
  if(m_requiresClas && m_config.allowBlasCaching)
  {
    resetCachedBlas(uploader);
  }
  if(m_requiresClas && m_config.usePersistentClasAllocator)
  {
    m_clasAllocator.cmdReset(uploader.getCmd());
  }
  uploader.flush();
}

bool SceneStreaming::initShadersAndPipelines()
{
  Resources& res = *m_resources;

  shaderc::CompileOptions options = res.makeCompilerOptions();
  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));

  shaderc::CompileOptions optionsRaster = options;
  optionsRaster.AddMacroDefinition("TARGETS_RASTERIZATION", "1");
  shaderc::CompileOptions optionsRay = options;
  optionsRay.AddMacroDefinition("TARGETS_RASTERIZATION", "0");

  res.compileShader(m_shaders.computeAgeFilterGroups, VK_SHADER_STAGE_COMPUTE_BIT, "stream_agefilter_groups.comp.glsl", &options);
  res.compileShader(m_shaders.computeSetup, VK_SHADER_STAGE_COMPUTE_BIT, "stream_setup.comp.glsl", &options);
  res.compileShader(m_shaders.computeUpdateSceneRaster, VK_SHADER_STAGE_COMPUTE_BIT, "stream_update_scene.comp.glsl", &optionsRaster);
  res.compileShader(m_shaders.computeUpdateSceneRay, VK_SHADER_STAGE_COMPUTE_BIT, "stream_update_scene.comp.glsl", &optionsRay);

  // we load all shaders regardless of use for now

  if(m_config.usePersistentClasAllocator)
  {
    res.compileShader(m_shaders.computeAllocatorBuildFreeGaps, VK_SHADER_STAGE_COMPUTE_BIT,
                      "stream_allocator_build_freegaps.comp.glsl", &options);
    res.compileShader(m_shaders.computeAllocatorFreeGapsInsert, VK_SHADER_STAGE_COMPUTE_BIT,
                      "stream_allocator_freegaps_insert.comp.glsl", &options);
    res.compileShader(m_shaders.computeAllocatorLoadGroups, VK_SHADER_STAGE_COMPUTE_BIT,
                      "stream_allocator_load_groups.comp.glsl", &options);
    res.compileShader(m_shaders.computeAllocatorSetupInsertion, VK_SHADER_STAGE_COMPUTE_BIT,
                      "stream_allocator_setup_insertion.comp.glsl", &options);
    res.compileShader(m_shaders.computeAllocatorUnloadGroups, VK_SHADER_STAGE_COMPUTE_BIT,
                      "stream_allocator_unload_groups.comp.glsl", &options);
  }
  else
  {
    res.compileShader(m_shaders.computeCompactionClasOld, VK_SHADER_STAGE_COMPUTE_BIT,
                      "stream_compaction_old_clas.comp.glsl", &options);
    res.compileShader(m_shaders.computeCompactionClasNew, VK_SHADER_STAGE_COMPUTE_BIT,
                      "stream_compaction_new_clas.comp.glsl", &options);
  }

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAgeFilterGroups);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAgeFilterGroups);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeSetup);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeUpdateSceneRaster);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeUpdateSceneRaster);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeUpdateSceneRay);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeUpdateSceneRay);

    if(m_config.usePersistentClasAllocator)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAllocatorBuildFreeGaps);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAllocatorBuildFreeGaps);

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAllocatorFreeGapsInsert);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAllocatorFreeGapsInsert);

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAllocatorLoadGroups);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAllocatorLoadGroups);

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAllocatorSetupInsertion);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAllocatorSetupInsertion);

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAllocatorUnloadGroups);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAllocatorUnloadGroups);
    }
    else
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeCompactionClasOld);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeCompactionClasOld);

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeCompactionClasNew);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeCompactionClasNew);
    }
  }

  return true;
}

void SceneStreaming::deinitShadersAndPipelines()
{
  Resources& res = *m_resources;

  res.destroyPipelines(m_pipelines);
}

bool SceneStreaming::initClas()
{
  // reset streaming for now, easier.
  // alternatively we could build CLAS for everything resident and adjust the max clas memory budget
  // if were to exceed it.
  reset();

  Resources& res            = *m_resources;
  m_stats.reservedClasBytes = m_config.maxClasMegaBytes * 1024 * 1024;
  m_clasOperationsSize      = 0;
  m_blasSize                = 0;

  m_requiresClas = true;

  VkPhysicalDeviceProperties2                              props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clusterProps = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
  props2.pNext = &clusterProps;
  vkGetPhysicalDeviceProperties2(res.m_physicalDevice, &props2);

  if(m_config.allowBlasCaching)
  {
    nvvk::BufferSubAllocator::InitInfo initInfo;
    initInfo.keepLastBlock    = false;
    initInfo.debugName        = "CachedBlasAllocator";
    initInfo.maxAllocatedSize = m_config.maxBlasCachingMegaBytes * 1024 * 1024;
    initInfo.blockSize        = std::min(size_t(16), m_config.maxBlasCachingMegaBytes) * 1024 * 1024;
    initInfo.memoryUsage      = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    initInfo.usageFlags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    initInfo.resourceAllocator = &res.m_allocator;
    initInfo.minAlignment      = clusterProps.clusterBottomLevelByteAlignment;
    m_cachedBlasAlignment      = initInfo.minAlignment;

    m_cachedBlasAllocator.init(initInfo);
  }

  m_clasScratchAlignment = clusterProps.clusterScratchByteAlignment;

  uint32_t maxNewPerFrameClusters = m_scene->m_config.clusterGroupSize * m_config.maxPerFrameLoadRequests;

  // setup update related data
  m_updates.initClas(res, m_config, m_scene->m_config);
  m_clasOperationsSize += logMemoryUsage(m_updates.getClasOperationsSize(), "operations", "stream clas updates");

  m_clasTriangleInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  m_clasTriangleInput.maxClusterTriangleCount       = m_scene->m_maxClusterTriangles;
  m_clasTriangleInput.maxClusterVertexCount         = m_scene->m_maxClusterVertices;
  m_clasTriangleInput.maxClusterUniqueGeometryCount = 1;
  m_clasTriangleInput.maxGeometryIndexValue         = 0;
  m_clasTriangleInput.minPositionTruncateBitCount =
      std::max(m_config.clasPositionTruncateBits,
               m_scene->m_config.useCompressedData ? uint32_t(m_scene->m_config.compressionPosDropBits) : 0);
  m_clasTriangleInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;

  {
    VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    VkClusterAccelerationStructureInputInfoNV clasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    // we need to know the size of certain operations

    // implicit newly build per frame
    clasInput.maxAccelerationStructureCount   = maxNewPerFrameClusters;
    clasInput.flags                           = m_config.clasBuildFlags;
    clasInput.opType                          = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    clasInput.opMode                          = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    clasInput.opInput.pTriangleClusters       = &m_clasTriangleInput;
    m_clasTriangleInput.maxTotalTriangleCount = m_scene->m_maxClusterTriangles * maxNewPerFrameClusters;
    m_clasTriangleInput.maxTotalVertexCount   = m_scene->m_maxClusterVertices * maxNewPerFrameClusters;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &clasInput, &buildSizesInfo);
    m_clasScratchNewBuildSize = buildSizesInfo.buildScratchSize;
    m_clasScratchNewClasSize  = buildSizesInfo.accelerationStructureSize;
    // we put scratch operation space after new clas data, make sure it has the proper alignment
    m_clasScratchNewClasSize = nvutils::align_up(m_clasScratchNewClasSize, m_clasScratchAlignment);

    // explicit build of single to get worst-case size
    clasInput.maxAccelerationStructureCount   = 1;
    clasInput.flags                           = m_config.clasBuildFlags;
    clasInput.opType                          = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    clasInput.opMode                          = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    clasInput.opInput.pTriangleClusters       = &m_clasTriangleInput;
    m_clasTriangleInput.maxTotalTriangleCount = m_scene->m_maxClusterTriangles;
    m_clasTriangleInput.maxTotalVertexCount   = m_scene->m_maxClusterVertices;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &clasInput, &buildSizesInfo);
    m_clasSingleMaxSize = buildSizesInfo.accelerationStructureSize;

    // move
    VkClusterAccelerationStructureMoveObjectsInputNV clasMoveInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV};
    clasMoveInput.type = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV;

    if(m_config.usePersistentClasAllocator)
    {
      // only need to move newly built into persistent storage once
      clasInput.maxAccelerationStructureCount = maxNewPerFrameClusters;
      clasMoveInput.maxMovedBytes             = m_clasScratchNewClasSize;
      // old clusters can overlap themselves
      clasMoveInput.noMoveOverlap = VK_TRUE;
    }
    else
    {
      // need to potentially move everything due to compaction of old clas prior adding new
      clasInput.maxAccelerationStructureCount = m_config.maxClusters;
      clasMoveInput.maxMovedBytes             = m_config.maxClasMegaBytes * 1024 * 1024;
      // old clusters can overlap themselves
      clasMoveInput.noMoveOverlap = VK_FALSE;
    }

    clasInput.flags                = 0;
    clasInput.opType               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
    clasInput.opMode               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    clasInput.opInput.pMoveObjects = &clasMoveInput;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &clasInput, &buildSizesInfo);
    m_clasScratchMoveSize = buildSizesInfo.updateScratchSize;

    m_clasScratchTotalSize = m_clasScratchNewClasSize + std::max(m_clasScratchMoveSize, m_clasScratchNewBuildSize);
  }


  if(m_config.usePersistentClasAllocator)
  {
    // setup the gpu-side memory manage for persistent clas storage
    m_clasAllocator.init(res, m_config.maxClasMegaBytes, uint32_t(m_clasSingleMaxSize) * m_scene->m_config.clusterGroupSize,
                         clusterProps.clusterByteAlignment << m_config.clasAllocatorGranularityShift,
                         m_config.clasAllocatorSectorSizeShift, m_shaderData.clasAllocator);
    m_clasOperationsSize += logMemoryUsage(m_clasAllocator.getOperationsSize(), "opertions", "clas alloc");

    m_stats.maxSizedReserved = m_clasAllocator.getMaxSized();
  }
  else
  {
    m_stats.maxSizedReserved = uint32_t(m_stats.reservedClasBytes / (m_clasSingleMaxSize * m_scene->m_config.clusterGroupSize));
  }

  {
    uint32_t                        loGroupsCount           = 0;
    uint32_t                        loClustersCount         = 0;
    uint32_t                        loMaxGroupClustersCount = 0;
    const StreamingResident::Group* groups =
        m_resident.initClas(res, m_config, m_shaderData.resident, loGroupsCount, loClustersCount, loMaxGroupClustersCount);

    assert(loGroupsCount == uint32_t(m_scene->getActiveGeometryCount()));

    m_clasOperationsSize += logMemoryUsage(m_resident.getClasOperationsSize(), "operations", "stream clas resident");

    size_t scratchSize = 0;
    size_t blasSize    = 0;

    VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    VkClusterAccelerationStructureInputInfoNV clasInputInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    clasInputInfo.maxAccelerationStructureCount = loClustersCount;
    clasInputInfo.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    clasInputInfo.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    clasInputInfo.opInput.pTriangleClusters     = &m_clasTriangleInput;
    m_clasTriangleInput.maxTotalTriangleCount   = m_scene->m_maxClusterTriangles * loClustersCount;
    m_clasTriangleInput.maxTotalVertexCount     = m_scene->m_maxClusterVertices * loClustersCount;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &clasInputInfo, &buildSizesInfo);
    scratchSize = std::max(scratchSize, buildSizesInfo.buildScratchSize);

    clasInputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &clasInputInfo, &buildSizesInfo);
    scratchSize = std::max(scratchSize, buildSizesInfo.buildScratchSize);


    VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    // low detail blas has only one cluster per blas
    blasInput.maxClusterCountPerAccelerationStructure = loMaxGroupClustersCount;
    blasInput.maxTotalClusterCount                    = loClustersCount;

    VkClusterAccelerationStructureInputInfoNV blasInputInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    blasInputInfo.flags  = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    blasInputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    blasInputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    blasInputInfo.opInput.pClustersBottomLevel  = &blasInput;
    blasInputInfo.maxAccelerationStructureCount = loGroupsCount;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &blasInputInfo, &buildSizesInfo);
    scratchSize = std::max(scratchSize, buildSizesInfo.buildScratchSize);
    blasSize    = buildSizesInfo.accelerationStructureSize;


    nvvk::Buffer scratchTemp;
    res.createBuffer(scratchTemp, scratchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    nvvk::BufferTyped<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> clasBuildInfosHost;
    res.createBufferTyped(clasBuildInfosHost, loClustersCount,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          VMA_MEMORY_USAGE_CPU_ONLY, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    nvvk::BufferTyped<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> blasBuildInfosHost;
    res.createBufferTyped(blasBuildInfosHost, loGroupsCount,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          VMA_MEMORY_USAGE_CPU_ONLY,
                          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    nvvk::BufferTyped<uint32_t> buildSizesHost;
    res.createBufferTyped(buildSizesHost, std::max(loClustersCount, loGroupsCount),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    nvvk::BufferTyped<uint64_t> buildAddressesHost;
    res.createBufferTyped(buildAddressesHost, std::max(loClustersCount, loGroupsCount),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);


    uint32_t                                                      clusterOffset  = 0;
    VkClusterAccelerationStructureBuildTriangleClusterInfoNV*     clasBuildInfos = clasBuildInfosHost.data();
    VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV* blasBuildInfos = blasBuildInfosHost.data();

    // prepare build of clusters
    for(uint32_t g = 0; g < loGroupsCount; g++)
    {
      const StreamingResident::Group& residentGroup = groups[g];
      const Scene::GeometryView& sceneGeometry  = m_scene->getActiveGeometry(residentGroup.geometryGroup.geometryID);
      const Scene::GroupInfo     sceneGroupInfo = sceneGeometry.groupInfos[residentGroup.geometryGroup.groupID];
      Scene::GroupView           sceneGroupView(sceneGeometry.groupData, sceneGroupInfo);

      uint64_t groupVA     = residentGroup.deviceAddress;
      size_t   indexOffset = size_t(sceneGroupView.indices.data()) - size_t(sceneGroupView.raw);

      blasBuildInfos[g].clusterReferencesCount  = residentGroup.clusterCount;
      blasBuildInfos[g].clusterReferencesStride = sizeof(uint64_t);
      blasBuildInfos[g].clusterReferences = m_shaderData.resident.clasAddresses + sizeof(uint64_t) * clusterOffset;


      for(uint32_t c = 0; c < residentGroup.clusterCount; c++)
      {
        const shaderio::Cluster& sceneCluster = sceneGroupView.clusters[c];
        assert((residentGroup.clusterResidentID + c) == clusterOffset);

        VkClusterAccelerationStructureBuildTriangleClusterInfoNV& buildInfo = clasBuildInfos[clusterOffset];
        buildInfo                                                           = {};

        uint64_t clusterVA = groupVA + sizeof(shaderio::Group) + sizeof(shaderio::Cluster) * c;

        buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
        buildInfo.clusterID         = residentGroup.clusterResidentID + c;
        buildInfo.triangleCount     = sceneCluster.triangleCountMinusOne + 1;
        buildInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
        buildInfo.indexBufferStride = 1;
        if(sceneGroupInfo.uncompressedSizeBytes)
        {
          buildInfo.indexBuffer = groupVA + indexOffset;
          indexOffset += buildInfo.triangleCount * 3;
        }
        else
        {
          buildInfo.indexBuffer = clusterVA + sceneCluster.indices;
        }

        buildInfo.vertexCount              = sceneCluster.vertexCountMinusOne + 1;
        buildInfo.vertexBufferStride       = uint16_t(sizeof(glm::vec3));
        buildInfo.vertexBuffer             = clusterVA + sceneCluster.vertices;
        buildInfo.positionTruncateBitCount = m_clasTriangleInput.minPositionTruncateBitCount;

        clusterOffset++;
      }
    }

    assert(clusterOffset == loClustersCount);

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.scratchData = scratchTemp.address;
    assert(cmdInfo.scratchData % m_clasScratchAlignment == 0);

    // first run is gather sizes

    cmdInfo.input        = clasInputInfo;
    cmdInfo.input.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;

    cmdInfo.srcInfosArray.deviceAddress = clasBuildInfosHost.address;
    cmdInfo.srcInfosArray.size          = clasBuildInfosHost.bufferSize;
    cmdInfo.srcInfosArray.stride        = sizeof(shaderio::ClasBuildInfo);

    cmdInfo.dstSizesArray.deviceAddress = buildSizesHost.address;
    cmdInfo.dstSizesArray.size          = buildSizesHost.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    VkCommandBuffer cmd = res.createTempCmdBuffer();
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
    res.tempSyncSubmit(cmd);

    // compute size, storage of lo-res geometry and destination addresses etc.

    size_t          clasSize         = 0;
    const uint32_t* clasSizesMapping = buildSizesHost.data();
    for(uint32_t c = 0; c < loClustersCount; c++)
    {
      clasSize += clasSizesMapping[c];
    }

    res.createBuffer(m_clasLowDetailBuffer, clasSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    NVVK_DBG_NAME(m_clasLowDetailBuffer.buffer);
    m_clasLowDetailSize = clasSize;

    clasSize                       = 0;
    uint64_t* clasAddressesMapping = buildAddressesHost.data();
    for(uint32_t c = 0; c < loClustersCount; c++)
    {
      clasAddressesMapping[c] = m_clasLowDetailBuffer.address + clasSize;
      clasSize += clasSizesMapping[c];
    }

    // second run is build explicit

    cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

    cmdInfo.dstAddressesArray.deviceAddress = buildAddressesHost.address;
    cmdInfo.dstAddressesArray.size          = buildAddressesHost.bufferSize;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmd = res.createTempCmdBuffer();
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    // and also copy the sizes & addresses to the persistent resident table

    const nvvk::Buffer& residentClasBuffer = m_resident.getClasBuffer();

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = m_shaderData.resident.clasSizes - residentClasBuffer.address;
    region.size      = buildSizesHost.bufferSize;

    vkCmdCopyBuffer(cmd, buildSizesHost.buffer, residentClasBuffer.buffer, 1, &region);

    region.srcOffset = 0;
    region.dstOffset = m_shaderData.resident.clasAddresses - residentClasBuffer.address;
    region.size      = buildAddressesHost.bufferSize;

    vkCmdCopyBuffer(cmd, buildAddressesHost.buffer, residentClasBuffer.buffer, 1, &region);

    if(m_config.usePersistentClasAllocator)
    {
      m_clasAllocator.cmdReset(cmd);
    }

    {
      // barrier
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

      // build low detail blas, one per low detail group
      res.createBuffer(m_clasLowDetailBlasBuffer, blasSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
      NVVK_DBG_NAME(m_clasLowDetailBlasBuffer.buffer);
      m_blasSize += blasSize;

      cmdInfo.input        = blasInputInfo;
      cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;

      cmdInfo.dstImplicitData = m_clasLowDetailBlasBuffer.address;

      cmdInfo.srcInfosArray.deviceAddress = blasBuildInfosHost.address;
      cmdInfo.srcInfosArray.size          = blasBuildInfosHost.bufferSize;
      cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

      cmdInfo.dstSizesArray.deviceAddress = buildSizesHost.address;
      cmdInfo.dstSizesArray.size          = buildSizesHost.bufferSize;
      cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

      cmdInfo.dstAddressesArray.deviceAddress = buildAddressesHost.address;
      cmdInfo.dstAddressesArray.size          = buildAddressesHost.bufferSize;
      cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
    }

    res.tempSyncSubmit(cmd);

    const uint64_t* blasAddresses = buildAddressesHost.data();

    for(uint32_t g = 0; g < loGroupsCount; g++)
    {
      const StreamingResident::Group& residentGroup  = groups[g];
      shaderio::Geometry&             shaderGeometry = m_shaderGeometries[residentGroup.geometryGroup.geometryID];

      shaderGeometry.lowDetailBlasAddress = blasAddresses[g];
    }

    res.simpleUploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

    res.m_allocator.destroyBuffer(scratchTemp);
    res.m_allocator.destroyBuffer(buildSizesHost);
    res.m_allocator.destroyBuffer(buildAddressesHost);
    res.m_allocator.destroyBuffer(clasBuildInfosHost);
    res.m_allocator.destroyBuffer(blasBuildInfosHost);
  }

  return true;
}

void SceneStreaming::deinitClas()
{
  // No reset is required, we just destroy all clas related resources.
  // What was fitting so far, is guaranteed to fit still.

  Resources& res = *m_resources;

  m_resident.deinitClas(res);
  m_updates.deinitClas(res);
  if(m_config.usePersistentClasAllocator)
  {
    m_clasAllocator.deinit(res);
  }
  m_shaderData.clasAllocator = {};

  res.m_allocator.destroyBuffer(m_clasLowDetailBuffer);
  res.m_allocator.destroyBuffer(m_clasLowDetailBlasBuffer);
  m_stats.reservedClasBytes = 0;

  if(m_config.allowBlasCaching)
  {
    for(auto& persistentGeometry : m_persistentGeometries)
    {
      if(persistentGeometry.cachedBlasAllocation)
      {
        m_cachedBlasAllocator.subFree(persistentGeometry.cachedBlasAllocation);
      }
    }

    m_cachedBlasAllocator.deinit();
  }

  for(auto& shaderGeometry : m_shaderGeometries)
  {
    shaderGeometry.lowDetailBlasAddress = 0;
  }
  res.simpleUploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  m_clasOperationsSize      = 0;
  m_clasLowDetailSize       = 0;
  m_clasSingleMaxSize       = 0;
  m_clasScratchNewClasSize  = 0;
  m_clasScratchNewBuildSize = 0;
  m_clasScratchMoveSize     = 0;
  m_clasScratchTotalSize    = 0;
  m_blasSize                = 0;

  m_requiresClas = false;
}

}  // namespace lodclusters
