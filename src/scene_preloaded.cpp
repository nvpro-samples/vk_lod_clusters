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
#include "scene_preloaded.hpp"

namespace lodclusters {

bool ScenePreloaded::canPreload(VkDeviceSize deviceLocalHeapSize, const Scene* scene)
{
  VkDeviceSize sizeLimit = (deviceLocalHeapSize * 600) / 1000;
  VkDeviceSize testSize  = 0;

  for(size_t geometryIndex = 0; geometryIndex < scene->getActiveGeometryCount(); geometryIndex++)
  {
    const Scene::GeometryView& sceneGeometry = scene->getActiveGeometry(geometryIndex);
    ScenePreloaded::Geometry   preloadGeometry;

    // * for CLAS estimate
    testSize += sceneGeometry.groupData.size() * 2;

    size_t numNodes = sceneGeometry.lodNodes.size();
    testSize += preloadGeometry.lodNodes.value_size * numNodes;
    testSize += preloadGeometry.lodNodeBboxes.value_size * numNodes;

    uint32_t numLodLevels = sceneGeometry.lodLevelsCount;
    testSize += preloadGeometry.lodLevels.value_size * numLodLevels;
  }

  if(testSize > sizeLimit)
  {
    LOGI("Likely exceeding device memory limit for preloaded scene\n");
    return false;
  }

  return true;
}

bool ScenePreloaded::init(Resources* res, const Scene* scene, const Config& config)
{
  assert(m_resources == nullptr && "init called without prior deinit");

  m_resources = res;
  m_scene     = scene;
  m_config    = config;

  if(!canPreload(res->getDeviceLocalHeapSize(), scene))
  {
    LOGW("Likely exceeding device memory limit for preloaded scene\n");
    return false;
  }


  m_shaderGeometries.resize(scene->getActiveGeometryCount());
  m_geometries.resize(scene->getActiveGeometryCount());

  Resources::BatchedUploader uploader(*res);

  uint32_t instancesOffset = 0;
  for(size_t geometryIndex = 0; geometryIndex < scene->getActiveGeometryCount(); geometryIndex++)
  {
    shaderio::Geometry&        shaderGeometry  = m_shaderGeometries[geometryIndex];
    ScenePreloaded::Geometry&  preloadGeometry = m_geometries[geometryIndex];
    const Scene::GeometryView& sceneGeometry   = scene->getActiveGeometry(geometryIndex);

    size_t groupDataSize = sceneGeometry.groupData.size_bytes();

    if(scene->m_config.useCompressedData)
    {
      groupDataSize = 0;
      for(size_t g = 0; g < sceneGeometry.groupInfos.size(); g++)
      {
        const Scene::GroupInfo groupInfo = sceneGeometry.groupInfos[g];
        groupDataSize += groupInfo.getDeviceSize();
      }
    }

    res->createBuffer(preloadGeometry.groupData, groupDataSize,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    NVVK_DBG_NAME(preloadGeometry.groupData.buffer);

    res->createBufferTyped(preloadGeometry.groupAddresses, sceneGeometry.groupInfos.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    res->createBufferTyped(preloadGeometry.clusterAddresses, sceneGeometry.totalClustersCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(preloadGeometry.groupAddresses.buffer);
    NVVK_DBG_NAME(preloadGeometry.clusterAddresses.buffer);

    size_t numNodes = sceneGeometry.lodNodes.size();
    res->createBufferTyped(preloadGeometry.lodNodes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    res->createBufferTyped(preloadGeometry.lodNodeBboxes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(preloadGeometry.lodNodes.buffer);
    NVVK_DBG_NAME(preloadGeometry.lodNodeBboxes.buffer);

    uint32_t numLodLevels = sceneGeometry.lodLevelsCount;
    res->createBufferTyped(preloadGeometry.lodLevels, numLodLevels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(preloadGeometry.lodLevels.buffer);

    m_geometrySize += preloadGeometry.groupData.bufferSize;
    m_geometrySize += preloadGeometry.groupAddresses.bufferSize;
    m_geometrySize += preloadGeometry.clusterAddresses.bufferSize;
    m_geometrySize += preloadGeometry.lodLevels.bufferSize;
    m_geometrySize += preloadGeometry.lodNodes.bufferSize;
    m_geometrySize += preloadGeometry.lodNodeBboxes.bufferSize;

    // setup shaderio
    shaderGeometry                    = {};
    shaderGeometry.bbox               = sceneGeometry.bbox;
    shaderGeometry.nodes              = preloadGeometry.lodNodes.address;
    shaderGeometry.nodeBboxes         = preloadGeometry.lodNodeBboxes.address;
    shaderGeometry.preloadedGroups    = preloadGeometry.groupAddresses.address;
    shaderGeometry.preloadedClusters  = preloadGeometry.clusterAddresses.address;
    shaderGeometry.lodLevelsCount     = uint32_t(numLodLevels);
    shaderGeometry.lodLevels          = preloadGeometry.lodLevels.address;
    shaderGeometry.cachedBlasAddress  = 0;
    shaderGeometry.cachedBlasLodLevel = TRAVERSAL_INVALID_LOD_LEVEL;
    shaderGeometry.instancesCount     = sceneGeometry.instanceReferenceCount * scene->getGeometryInstanceFactor();
    shaderGeometry.instancesOffset    = instancesOffset;

    instancesOffset += shaderGeometry.instancesCount;

    // lowest detail group must have just a single cluster
    shaderio::LodLevel lastLodLevel = sceneGeometry.lodLevels.back();
    assert(lastLodLevel.groupCount == 1 && lastLodLevel.clusterCount == 1);

    shaderGeometry.lowDetailClusterID = lastLodLevel.clusterOffset;
    shaderGeometry.lowDetailTriangles = sceneGeometry.groupInfos[lastLodLevel.groupOffset].triangleCount;

    // basic uploads

    uploader.uploadBuffer(preloadGeometry.lodNodes, sceneGeometry.lodNodes.data());
    uploader.uploadBuffer(preloadGeometry.lodNodeBboxes, sceneGeometry.lodNodeBboxes.data());
    uploader.uploadBuffer(preloadGeometry.lodLevels, sceneGeometry.lodLevels.data());

    // clusters and groups need to be filled manually

    uint64_t* clusterAddresses = uploader.uploadBuffer(preloadGeometry.clusterAddresses, (uint64_t*)nullptr);
    uint64_t* groupAddresses =
        uploader.uploadBuffer(preloadGeometry.groupAddresses, (uint64_t*)nullptr, Resources::FlushState::DONT_FLUSH);
    uint8_t* groupData = uploader.uploadBuffer(preloadGeometry.groupData, (uint8_t*)nullptr, Resources::FlushState::DONT_FLUSH);

    uint32_t clusterOffset   = 0;
    size_t   groupDataOffset = 0;
    for(size_t g = 0; g < sceneGeometry.groupInfos.size(); g++)
    {
      const Scene::GroupInfo groupInfo = sceneGeometry.groupInfos[g];
      const Scene::GroupView groupView(sceneGeometry.groupData, groupInfo);
      uint64_t               groupVA = preloadGeometry.groupData.address + groupDataOffset;

      groupAddresses[g] = groupVA;

      Scene::fillGroupRuntimeData(groupInfo, groupView, uint32_t(g), uint32_t(g), clusterOffset,
                                  groupData + groupDataOffset, groupInfo.getDeviceSize());

      groupDataOffset += groupInfo.getDeviceSize();

      for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
      {
        clusterAddresses[c + clusterOffset] = groupVA + sizeof(shaderio::Group) + sizeof(shaderio::Cluster) * c;
      }

      clusterOffset += groupInfo.clusterCount;
    }
  }

  res->createBufferTyped(m_shaderGeometriesBuffer, scene->getActiveGeometryCount(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NVVK_DBG_NAME(m_shaderGeometriesBuffer.buffer);
  m_operationsSize += logMemoryUsage(m_shaderGeometriesBuffer.bufferSize, "operations", "preloaded geo buffer");

  uploader.uploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());
  uploader.flush();

  return true;
}

bool ScenePreloaded::updateClasRequired(bool state)
{
  if(state != m_hasClas)
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

void ScenePreloaded::deinit()
{
  if(!m_resources)
    return;

  for(auto& it : m_geometries)
  {
    m_resources->m_allocator.destroyBuffer(it.clusterAddresses);
    m_resources->m_allocator.destroyBuffer(it.groupData);
    m_resources->m_allocator.destroyBuffer(it.groupAddresses);
    m_resources->m_allocator.destroyBuffer(it.lodNodes);
    m_resources->m_allocator.destroyBuffer(it.lodNodeBboxes);
    m_resources->m_allocator.destroyBuffer(it.lodLevels);
    m_resources->m_allocator.destroyBuffer(it.clasData);
    m_resources->m_allocator.destroyBuffer(it.clusterClasAddresses);
    m_resources->m_allocator.destroyBuffer(it.clusterClasSizes);
  }

  m_resources->m_allocator.destroyBuffer(m_clasLowDetailBlasBuffer);

  m_resources->m_allocator.destroyBuffer(m_shaderGeometriesBuffer);
  m_resources    = nullptr;
  m_scene        = nullptr;
  m_geometrySize = 0;
}

bool ScenePreloaded::initClas()
{
  Resources* res = m_resources;
  m_hasClas      = true;

  m_clasOperationsSize = 0;
  m_clasSize           = 0;
  m_blasSize           = 0;

  VkClusterAccelerationStructureTriangleClusterInputNV clusterTriangleInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  clusterTriangleInput.maxClusterTriangleCount       = m_scene->m_maxClusterTriangles;
  clusterTriangleInput.maxClusterVertexCount         = m_scene->m_maxClusterVertices;
  clusterTriangleInput.maxClusterUniqueGeometryCount = 1;
  clusterTriangleInput.maxGeometryIndexValue         = 0;
  clusterTriangleInput.minPositionTruncateBitCount =
      std::max(m_config.clasPositionTruncateBits,
               m_scene->m_config.useCompressedData ? uint32_t(m_scene->m_config.compressionPosDropBits) : 0);
  clusterTriangleInput.maxTotalTriangleCount = m_scene->m_maxPerGeometryTriangles;
  clusterTriangleInput.maxTotalVertexCount   = m_scene->m_maxPerGeometryVertices;
  clusterTriangleInput.vertexFormat          = VK_FORMAT_R32G32B32_SFLOAT;

  VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
  // low detail blas has only one cluster per blas
  blasInput.maxClusterCountPerAccelerationStructure = 1;
  blasInput.maxTotalClusterCount                    = uint32_t(m_scene->getActiveGeometryCount());

  VkDeviceSize scratchSize = 0;
  VkDeviceSize blasSize    = 0;

  VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  VkClusterAccelerationStructureInputInfoNV clusterInputInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  clusterInputInfo.flags                         = m_config.clasBuildFlags;
  clusterInputInfo.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  clusterInputInfo.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
  clusterInputInfo.opInput.pTriangleClusters     = &clusterTriangleInput;
  clusterInputInfo.maxAccelerationStructureCount = m_scene->m_maxPerGeometryClusters;
  vkGetClusterAccelerationStructureBuildSizesNV(res->m_device, &clusterInputInfo, &buildSizesInfo);
  scratchSize = std::max(scratchSize, buildSizesInfo.buildScratchSize);

  VkClusterAccelerationStructureInputInfoNV blasInputInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  blasInputInfo.flags                        = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  blasInputInfo.opMode                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  blasInputInfo.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  blasInputInfo.opInput.pClustersBottomLevel = &blasInput;
  blasInputInfo.maxAccelerationStructureCount = uint32_t(m_scene->getActiveGeometryCount());
  vkGetClusterAccelerationStructureBuildSizesNV(res->m_device, &blasInputInfo, &buildSizesInfo);
  scratchSize = std::max(scratchSize, buildSizesInfo.buildScratchSize);
  blasSize    = buildSizesInfo.accelerationStructureSize;

  nvvk::Buffer scratchTemp;
  res->createBuffer(scratchTemp, scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  nvvk::BufferTyped<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> clasBuildInfosHost;
  res->createBufferTyped(clasBuildInfosHost, m_scene->m_maxPerGeometryClusters,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::BufferTyped<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> blasBuildInfosHost;
  res->createBufferTyped(blasBuildInfosHost, m_scene->getActiveGeometryCount(),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  nvvk::BufferTyped<uint32_t> buildSizesHost;
  res->createBufferTyped(buildSizesHost,
                         std::max(m_scene->m_maxPerGeometryClusters, uint32_t(m_scene->getActiveGeometryCount())),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  nvvk::BufferTyped<uint64_t> buildAddressesHost;
  res->createBufferTyped(buildAddressesHost,
                         std::max(m_scene->m_maxPerGeometryClusters, uint32_t(m_scene->getActiveGeometryCount())),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  VkCommandBuffer cmd;
  VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};

  for(size_t g = 0; g < m_scene->getActiveGeometryCount(); g++)
  {
    ScenePreloaded::Geometry&  preloadGeometry = m_geometries[g];
    shaderio::Geometry&        shaderGeometry  = m_shaderGeometries[g];
    const Scene::GeometryView& sceneGeometry   = m_scene->getActiveGeometry(g);

    VkClusterAccelerationStructureBuildTriangleClusterInfoNV* buildInfos = clasBuildInfosHost.data();

    size_t   groupOffset   = 0;
    uint32_t clusterOffset = 0;
    for(size_t g = 0; g < sceneGeometry.groupInfos.size(); g++)
    {
      const Scene::GroupInfo groupInfo = sceneGeometry.groupInfos[g];
      Scene::GroupView       groupView(sceneGeometry.groupData, groupInfo);
      uint64_t               groupVA = preloadGeometry.groupData.address + groupOffset;

      size_t indexOffset = size_t(groupView.indices.data()) - size_t(groupView.raw);

      for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
      {
        const shaderio::Cluster&                                  groupCluster = groupView.clusters[c];
        VkClusterAccelerationStructureBuildTriangleClusterInfoNV& buildInfo    = buildInfos[clusterOffset];
        buildInfo                                                              = {};

        uint64_t clusterVA = groupVA + sizeof(shaderio::Group) + sizeof(shaderio::Cluster) * c;

        buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
        buildInfo.clusterID         = clusterOffset;
        buildInfo.triangleCount     = groupCluster.triangleCountMinusOne + 1;
        buildInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
        buildInfo.indexBufferStride = 1;

        if(groupInfo.uncompressedSizeBytes)
        {
          buildInfo.indexBuffer = groupVA + indexOffset;
          indexOffset += buildInfo.triangleCount * 3;
        }
        else
        {
          buildInfo.indexBuffer = clusterVA + groupCluster.indices;
        }

        buildInfo.vertexCount              = groupCluster.vertexCountMinusOne + 1;
        buildInfo.vertexBufferStride       = uint16_t(sizeof(glm::vec3));
        buildInfo.vertexBuffer             = clusterVA + groupCluster.vertices;
        buildInfo.positionTruncateBitCount = clusterTriangleInput.minPositionTruncateBitCount;

        clusterOffset++;
      }

      groupOffset += groupInfo.getDeviceSize();
    }

    size_t numClusters = sceneGeometry.totalClustersCount;
    res->createBufferTyped(preloadGeometry.clusterClasAddresses, numClusters,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    NVVK_DBG_NAME(preloadGeometry.clusterClasAddresses.buffer);
    res->createBufferTyped(preloadGeometry.clusterClasSizes, numClusters,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    NVVK_DBG_NAME(preloadGeometry.clusterClasSizes.buffer);

    m_clasOperationsSize += preloadGeometry.clusterClasAddresses.bufferSize;
    m_clasOperationsSize += preloadGeometry.clusterClasSizes.bufferSize;

    // update shader visible pointers
    shaderGeometry.preloadedClusterClasAddresses = preloadGeometry.clusterClasAddresses.address;
    shaderGeometry.preloadedClusterClasSizes     = preloadGeometry.clusterClasSizes.address;

    clusterTriangleInput.maxTotalTriangleCount = sceneGeometry.totalTriangleCount;
    clusterTriangleInput.maxTotalVertexCount   = sceneGeometry.totalVerticesCount;

    clusterInputInfo.maxAccelerationStructureCount = sceneGeometry.totalClustersCount;
    clusterInputInfo.opInput.pTriangleClusters     = &clusterTriangleInput;

    cmdInfo.srcInfosArray.deviceAddress = clasBuildInfosHost.address;
    cmdInfo.srcInfosArray.size          = clasBuildInfosHost.bufferSize;
    cmdInfo.srcInfosArray.stride        = sizeof(shaderio::ClasBuildInfo);

    cmdInfo.dstSizesArray.deviceAddress = preloadGeometry.clusterClasSizes.address;
    cmdInfo.dstSizesArray.size          = preloadGeometry.clusterClasSizes.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.scratchData = scratchTemp.address;

    cmd = res->createTempCmdBuffer();
    {
      // build size mode only

      cmdInfo.input        = clusterInputInfo;
      cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;

      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

      // download sizes

      VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      memBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);

      VkBufferCopy region = {};
      region.size         = sizeof(uint32_t) * sceneGeometry.totalClustersCount;
      vkCmdCopyBuffer(cmd, preloadGeometry.clusterClasSizes.buffer, buildSizesHost.buffer, 1, &region);
    }
    res->tempSyncSubmit(cmd);

    // allocate clas data and setup per-clas offsets
    {
      uint64_t        sumClasSizes = 0;
      const uint32_t* clasSizes    = buildSizesHost.data();
      for(uint32_t c = 0; c < sceneGeometry.totalClustersCount; c++)
      {
        assert(clasSizes[c] && "clas with invalid size");
        sumClasSizes += clasSizes[c];
      }

      res->createBuffer(preloadGeometry.clasData, sumClasSizes, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
      NVVK_DBG_NAME(preloadGeometry.clasData.buffer);

      uint64_t  clasOffset    = 0;
      uint64_t* clasAddresses = buildAddressesHost.data();
      for(uint32_t c = 0; c < sceneGeometry.totalClustersCount; c++)
      {
        clasAddresses[c] = preloadGeometry.clasData.address + clasOffset;
        clasOffset += clasSizes[c];
      }
    }

    cmd = res->createTempCmdBuffer();
    {
      // upload addresses
      VkBufferCopy region = {};
      region.size         = sizeof(uint64_t) * sceneGeometry.totalClustersCount;
      vkCmdCopyBuffer(cmd, buildAddressesHost.buffer, preloadGeometry.clusterClasAddresses.buffer, 1, &region);

      VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
      memBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);

      // build for real

      cmdInfo.input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

      cmdInfo.dstAddressesArray.deviceAddress = preloadGeometry.clusterClasAddresses.address;
      cmdInfo.dstAddressesArray.size          = preloadGeometry.clusterClasAddresses.bufferSize;
      cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
    }
    res->tempSyncSubmit(cmd);


    {
      // setup blas build info
      VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV* blasInfos = blasBuildInfosHost.data();
      // just a single cluster
      blasInfos[g].clusterReferencesCount  = 1;
      blasInfos[g].clusterReferencesStride = uint32_t(sizeof(uint64_t));
      blasInfos[g].clusterReferences = preloadGeometry.clusterClasAddresses.addressAt(shaderGeometry.lowDetailClusterID);
    }

    m_clasSize += preloadGeometry.clasData.bufferSize;
  }

  logMemoryUsage(m_clasOperationsSize, "operations", "preloaded clas");

  // create low detail blas
  {
    res->createBuffer(m_clasLowDetailBlasBuffer, blasSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    NVVK_DBG_NAME(m_clasLowDetailBlasBuffer.buffer);
    m_blasSize = blasSize;

    cmd = res->createTempCmdBuffer();

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

    res->tempSyncSubmit(cmd);

    const uint64_t* blasAddresses = buildAddressesHost.data();

    for(size_t g = 0; g < m_scene->getActiveGeometryCount(); g++)
    {
      ScenePreloaded::Geometry& preloadGeometry = m_geometries[g];
      shaderio::Geometry&       shaderGeometry  = m_shaderGeometries[g];

      shaderGeometry.lowDetailBlasAddress = blasAddresses[g];
    }
  }

  m_resources->simpleUploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  res->m_allocator.destroyBuffer(scratchTemp);
  res->m_allocator.destroyBuffer(buildSizesHost);
  res->m_allocator.destroyBuffer(buildAddressesHost);
  res->m_allocator.destroyBuffer(clasBuildInfosHost);
  res->m_allocator.destroyBuffer(blasBuildInfosHost);

  return true;
}

void ScenePreloaded::deinitClas()
{
  for(size_t g = 0; g < m_geometries.size(); g++)
  {
    ScenePreloaded::Geometry& preloadGeometry = m_geometries[g];
    shaderio::Geometry&       shaderGeometry  = m_shaderGeometries[g];
    m_resources->m_allocator.destroyBuffer(preloadGeometry.clusterClasAddresses);
    m_resources->m_allocator.destroyBuffer(preloadGeometry.clusterClasSizes);
    m_resources->m_allocator.destroyBuffer(preloadGeometry.clasData);
    shaderGeometry.preloadedClusterClasAddresses = 0;
    shaderGeometry.preloadedClusterClasSizes     = 0;
    shaderGeometry.lowDetailBlasAddress          = 0;
  }

  m_resources->m_allocator.destroyBuffer(m_clasLowDetailBlasBuffer);

  m_resources->simpleUploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  m_clasSize           = 0;
  m_clasOperationsSize = 0;
  m_blasSize           = 0;
  m_hasClas            = false;
}
}  // namespace lodclusters
