
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <glm/gtc/constants.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <meshoptimizer.h>

#include "scene.hpp"
#include "threadlocal_arena.hpp"
#include "../shaders/attribute_encoding.h"

namespace lodclusters {

template <typename T0, typename T1>
void padZeroes(std::span<T0>& previous, T1* next)
{
  // fill padded array with zero
  {
    size_t padSize = size_t(next) - size_t(previous.data() + previous.size());
    if(padSize)
    {
      memset(previous.data() + previous.size(), 0, padSize);
    }
  }
}

// Number of float entries a cluster's attribute block (normals/tangents + texcoords) occupies in the
// runtime "[attributes][positions]" vertex-region layout (positions live in a separate trailing region).
// Texcoords are kept 8-byte (2-float) aligned relative to the block start.
static inline uint32_t clusterSeparatedAttrFloats(uint32_t vertexCount, uint32_t attributeBits)
{
  uint32_t n = (attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL) ? vertexCount : 0;
  if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
  {
    n = (n + 1u) & ~1u;
    n += 2u * vertexCount;
  }
  if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1)
  {
    n = (n + 1u) & ~1u;
    n += 2u * vertexCount;
  }
  return n;
}

void Scene::applyMaterialStateBits(uint32_t& stateBits, uint32_t clusterBits, bool isFirst)
{
  if(!isFirst)
  {
    if((stateBits & shaderio::CLUSTER_STATE_ALPHAMASKED) != (clusterBits & shaderio::CLUSTER_STATE_ALPHAMASKED))
      stateBits |= shaderio::CLUSTER_STATE_ALPHAMASKED_MIXED;
    if((stateBits & shaderio::CLUSTER_STATE_TWOSIDED) != (clusterBits & shaderio::CLUSTER_STATE_TWOSIDED))
      stateBits |= shaderio::CLUSTER_STATE_TWOSIDED_MIXED;
  }

  stateBits |= clusterBits;
}

void Scene::applyMaterialStateBits(uint32_t& stateBits, const GeometryStorage& geometry, uint32_t localMaterialID, bool isFirst)
{
  const Material& material = m_materials[geometry.localMaterialIDs[localMaterialID]];

  if(!isFirst)
  {
    if(material.alphaMasked != ((stateBits & shaderio::CLUSTER_STATE_ALPHAMASKED) != 0))
      stateBits |= shaderio::CLUSTER_STATE_ALPHAMASKED_MIXED;
    if(material.twoSided != ((stateBits & shaderio::CLUSTER_STATE_TWOSIDED) != 0))
      stateBits |= shaderio::CLUSTER_STATE_TWOSIDED_MIXED;
  }

  if(material.alphaMasked)
    stateBits |= shaderio::CLUSTER_STATE_ALPHAMASKED;
  if(material.twoSided)
    stateBits |= shaderio::CLUSTER_STATE_TWOSIDED;
}

void Scene::applyMaterialTriangleBits(uint8_t& triangleBits, const GeometryStorage& geometry, uint32_t localMaterialID)
{
  const Material& material = m_materials[geometry.localMaterialIDs[localMaterialID]];
  if(material.alphaMasked)
    triangleBits |= SHADERIO_CLUSTER_TRIANGLE_ALPHAMASKED;
  if(material.twoSided)
    triangleBits |= SHADERIO_CLUSTER_TRIANGLE_TWOSIDED;
}


uint8_t Scene::getMaterialLocalIndex(const GeometryStorage& geometry, uint32_t index, uint32_t attributeStride)
{
  return uint8_t(geometry.vertexAttributes[index * attributeStride + geometry.attributeMaterialOffset]);
}

// Takes the resulting cluster group of the lod generation and stores it into
// the internal representation used at runtime. This data is saved
// as is into the scene cache file and patched after upload when streamed in.
// Some abstraction is used to deal with results from either `meshoptimizer's` clusterlod,
// or `nv_cluster_lod_builder`.
uint32_t Scene::storeGroup(TempContext*       context,
                           uint32_t           threadIndex,
                           uint32_t           groupIndex,
                           const clodGroup&   group,
                           uint32_t           clusterCount,
                           const clodCluster* clusters)
{
  ProcessingInfo&  processing = context->processingInfo;
  GeometryStorage& geometry   = context->geometry;
  Scene::GroupInfo groupInfo  = {};

  uint32_t level = uint32_t(group.depth);

  uint8_t* groupTempData = &context->threadGroupDatas[context->threadGroupSize * threadIndex];

  Scene::GroupInfo groupTempInfo = context->threadGroupInfo;
  GroupStorage     groupTempStorage(groupTempData, groupTempInfo);

  std::span<uint32_t> vertexCacheEarlyValue((uint32_t*)(groupTempData + context->threadGroupStorageSize), 256);
  std::span<uint32_t> vertexCacheEarlyPos((uint32_t*)vertexCacheEarlyValue.data() + 256, 256);
  std::span<uint32_t> vertexCacheLocal(vertexCacheEarlyPos.data() + 256, m_config.clusterGroupSize * m_config.clusterVertices);

  uint32_t       clusterMaxVerticesCount  = 0;
  uint32_t       clusterMaxTrianglesCount = 0;
  shaderio::BBox groupBbox                = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  bool hasMultiMaterial = geometry.localMaterialIDs.size() > 1;

  // Union of CLUSTER_STATE_* across clusters (group-level material hints).
  uint32_t groupStateBits = 0;

  // Fill all data into temporary group storage.
  // This also does vertex de-duplication, prior that we don't know the final group storage
  // requirements in advance.
  //
  // After this pass we make the allocation request within a mutex in which we
  // mostly just copy out the data.

  {
    // running offsets
    uint32_t triangleOffset = 0;
    uint32_t vertexOffset   = 0;

    // runtime data offsets so all vertex/triangle data for a single cluster
    // is in a contiguous region
    uint32_t vertexDataOffset   = 0;
    uint32_t triangleDataOffset = 0;

    // stats
    size_t vertexPosBytes      = 0;
    size_t vertexNrmBytes      = 0;
    size_t vertexTexCoordBytes = 0;

    uint32_t attributeStride = uint32_t(geometry.vertexAttributes.size() / geometry.vertexPositions.size());

    for(uint32_t c = 0; c < clusterCount; c++)
    {
      uint32_t* localVertices = &vertexCacheLocal[vertexOffset];

      const clodCluster& tempCluster = clusters[c];

      shaderio::Cluster& groupCluster  = groupTempStorage.clusters[c];
      uint32_t           triangleCount = uint32_t(tempCluster.index_count / 3);
      uint32_t           vertexCount   = 0;
      uint32_t           stateBits     = 0;

      assert(triangleCount);

      groupCluster.localMaterialID = 0;
      // temp storage: store running float/byte indices; converted to packed cluster-relative
      // offsets during the final assembly below. positions slot = vertex-data float index,
      // triangles slot = triangle-data byte index.
      shaderio::Cluster_setOffsets(groupCluster, vertexDataOffset, 0, triangleDataOffset);


      // start assigning first triangle's material
      uint8_t localMaterialID = 0;
      if(hasMultiMaterial)
      {
        localMaterialID              = getMaterialLocalIndex(geometry, tempCluster.indices[0], attributeStride);
        groupCluster.localMaterialID = localMaterialID;
      }

      applyMaterialStateBits(stateBits, geometry, localMaterialID, true);

      memset(vertexCacheEarlyValue.data(), ~0, vertexCacheEarlyValue.size_bytes());
      for(uint32_t i = 0; i < tempCluster.index_count; i++)
      {
        uint32_t vertexIndex = tempCluster.indices[i];
        uint32_t cacheIndex  = ~0;

        // quick early out, have we seen the index
        uint32_t cacheEarlyValue = vertexCacheEarlyValue[vertexIndex & 0xFF];
        if(cacheEarlyValue == vertexIndex)
        {
          cacheIndex = vertexCacheEarlyPos[vertexIndex & 0xFF];
        }
        else
        {
          // look for it serially
          for(uint32_t v = 0; v < vertexCount; v++)
          {
            if(localVertices[v] == vertexIndex)
            {
              cacheIndex = v;
            }
          }
        }

        if(cacheIndex == ~0)
        {
          cacheIndex                                = vertexCount++;
          localVertices[cacheIndex]                 = vertexIndex;
          vertexCacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;
          vertexCacheEarlyPos[vertexIndex & 0xFF]   = cacheIndex;

          if(hasMultiMaterial && localMaterialID != getMaterialLocalIndex(geometry, vertexIndex, attributeStride))
          {
            // signals that we have per-triangle assignments
            groupCluster.localMaterialID = SHADERIO_PER_TRIANGLE_MATERIALS;
          }
        }
        groupTempStorage.triangles[i + triangleDataOffset] = uint8_t(cacheIndex);
      }
      triangleDataOffset += uint32_t(tempCluster.index_count);


      // build per-triangle material indices
      if(groupCluster.localMaterialID == SHADERIO_PER_TRIANGLE_MATERIALS)
      {
        processing.stats.multiMaterialClusters++;

        stateBits = 0;

        for(uint32_t t = 0; t < triangleCount; t++)
        {
          uint8_t vertexMaterial[3];
          vertexMaterial[0] = getMaterialLocalIndex(geometry, tempCluster.indices[t * 3 + 0], attributeStride);
          vertexMaterial[1] = getMaterialLocalIndex(geometry, tempCluster.indices[t * 3 + 1], attributeStride);
          vertexMaterial[2] = getMaterialLocalIndex(geometry, tempCluster.indices[t * 3 + 2], attributeStride);

          bool diff01 = vertexMaterial[0] != vertexMaterial[1];
          bool diff12 = vertexMaterial[1] != vertexMaterial[2];
          // not all are the same
          if(diff01 || diff12)
          {
            // let's pick where maybe at least 2 are the same
            bool diff02 = vertexMaterial[0] != vertexMaterial[2];
            if(diff02)
            {
              vertexMaterial[0] = vertexMaterial[2];
            }
          }
          applyMaterialStateBits(stateBits, geometry, vertexMaterial[0], t == 0);
          applyMaterialTriangleBits(vertexMaterial[0], geometry, vertexMaterial[0]);
          groupTempStorage.triangles[t + triangleDataOffset] = vertexMaterial[0];
        }

        triangleDataOffset += triangleCount;
      }

      shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, FLT_MAX, -FLT_MAX};
      {
        // in compression case we pack the attributes later
        if(m_config.useCompressedData)
        {
          for(uint32_t v = 0; v < vertexCount; v++)
          {
            glm::vec3 pos = geometry.vertexPositions[localVertices[v]];

            // local bbox
            bbox.lo = glm::min(bbox.lo, glm::vec3(pos));
            bbox.hi = glm::max(bbox.hi, glm::vec3(pos));
          }
        }
        else
        {
          for(uint32_t v = 0; v < vertexCount; v++)
          {
            // copy position
            glm::vec3 pos = geometry.vertexPositions[localVertices[v]];
            *(glm::vec3*)&groupTempStorage.vertices[vertexDataOffset + v * 3] = pos;

            // local bbox
            bbox.lo = glm::min(bbox.lo, glm::vec3(pos));
            bbox.hi = glm::max(bbox.hi, glm::vec3(pos));
          }
        }

        vertexDataOffset += vertexCount * 3;
        vertexPosBytes += sizeof(float) * 3 * vertexCount;
      }

      if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
      {
        // in compression case we pack attributes later
        if(!m_config.useCompressedData)
        {
          if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
          {
            for(uint32_t v = 0; v < vertexCount; v++)
            {
              glm::vec3 normal =
                  *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
              glm::vec4 tangent =
                  *(const glm::vec4*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeTangentOffset]);

              uint32_t encoded = shaderio::normal_pack(normal);
              encoded |= shaderio::tangent_pack(normal, tangent) << ATTRENC_NORMAL_BITS;

              *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
            }
          }
          else
          {
            for(uint32_t v = 0; v < vertexCount; v++)
            {
              glm::vec3 tmp =
                  *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);

              uint32_t encoded                                             = shaderio::normal_pack(tmp);
              *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
            }
          }
        }

        vertexDataOffset += vertexCount;
        vertexNrmBytes += sizeof(uint32_t) * vertexCount;
      }

      for(uint32_t t = 0; t < 2; t++)
      {
        shaderio::ClusterAttributeBits usedBit =
            t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
        uint32_t attributeTexOffset = t == 0 ? geometry.attributeTex0offset : geometry.attributeTex1offset;

        if(geometry.attributeBits & usedBit)
        {
          // align to vec2
          vertexDataOffset = (vertexDataOffset + 1) & ~1;

          if(!m_config.useCompressedData)
          {
            for(uint32_t v = 0; v < vertexCount; v++)
            {
              glm::vec2 tmp =
                  *(const glm::vec2*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + attributeTexOffset]);
              *(glm::vec2*)&groupTempStorage.vertices[vertexDataOffset + v * 2] = tmp;
            }
          }

          vertexDataOffset += vertexCount * 2;
          vertexTexCoordBytes += sizeof(float) * 2 * vertexCount;
        }
      }

      // find shortest and longest edge
      for(uint32_t t = 0; t < triangleCount; t++)
      {
        glm::vec3 trianglePositions[3];

        for(uint32_t v = 0; v < 3; v++)
        {
          trianglePositions[v] =
              geometry.vertexPositions[localVertices[groupTempStorage.triangles[(triangleOffset + t) * 3 + v]]];
        }

        for(uint32_t e = 0; e < 3; e++)
        {
          float distance    = glm::distance(trianglePositions[e], trianglePositions[(e + 1) % 3]);
          bbox.shortestEdge = std::min(bbox.shortestEdge, distance);
          bbox.longestEdge  = std::max(bbox.longestEdge, distance);
        }
      }

      groupBbox.lo = glm::min(groupBbox.lo, bbox.lo);
      groupBbox.hi = glm::max(groupBbox.hi, bbox.hi);

      groupTempStorage.clusterBboxes[c]           = bbox;
      groupTempStorage.clusterGeneratingGroups[c] = uint32_t(tempCluster.refined);

      groupCluster.triangleCountMinusOne = uint8_t(triangleCount - 1);
      groupCluster.vertexCountMinusOne   = uint8_t(vertexCount - 1);
      groupCluster.lodLevel              = uint8_t(level);
      groupCluster.groupChildIndex       = uint8_t(c);
      groupCluster.attributeBits         = uint8_t(geometry.attributeBits);
      groupCluster.stateBits             = uint8_t(stateBits);

      applyMaterialStateBits(groupStateBits, stateBits, c == 0);

      clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, triangleCount);
      clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, vertexCount);

      ((std::atomic_uint32_t&)m_histograms.clusterTriangles[triangleCount])++;
      ((std::atomic_uint32_t&)m_histograms.clusterVertices[vertexCount])++;

      vertexOffset += vertexCount;
      triangleOffset += triangleCount;
    }

    groupInfo.offsetBytes   = 0;
    groupInfo.clusterCount  = uint8_t(clusterCount);
    groupInfo.triangleCount = uint16_t(triangleOffset);
    groupInfo.vertexCount   = uint16_t(vertexOffset);
    groupInfo.lodLevel      = uint8_t(level);
    // runtime vertex region is laid out as [attributes][positions]; compute its float count
    {
      uint32_t attrRunning = 0;
      for(uint32_t c = 0; c < clusterCount; c++)
      {
        uint32_t v  = uint32_t(groupTempStorage.clusters[c].vertexCountMinusOne) + 1;
        attrRunning = (attrRunning + 1u) & ~1u;  // 2-float align each cluster's attribute block
        attrRunning += clusterSeparatedAttrFloats(v, groupTempStorage.clusters[c].attributeBits);
      }
      uint32_t attrTotal        = (attrRunning + 1u) & ~1u;  // positions region starts 2-float aligned
      groupInfo.vertexDataCount = attrTotal + 3u * vertexOffset;
    }
    groupInfo.triangleDataCount           = triangleDataOffset;
    groupInfo.uncompressedVertexDataCount = 0;
    groupInfo.uncompressedSizeBytes       = 0;
    groupInfo.sizeBytes                   = groupInfo.computeSize();

    {
      processing.stats.groups++;
      processing.stats.clusters += clusterCount;
      processing.stats.vertices += vertexOffset;
      processing.stats.groupHeaderBytes += sizeof(shaderio::Group);
      processing.stats.clusterHeaderBytes += sizeof(shaderio::Cluster) * clusterCount;
      processing.stats.clusterBboxBytes += sizeof(shaderio::BBox) * clusterCount;
      processing.stats.clusterGenBytes += sizeof(uint32_t) * clusterCount;
      processing.stats.triangleIndexBytes += sizeof(uint8_t) * triangleOffset * 3;
      processing.stats.triangleDataBytes += sizeof(uint8_t) * triangleDataOffset;
      processing.stats.vertexPosBytes += vertexPosBytes;
      processing.stats.vertexNrmBytes += vertexNrmBytes;
      processing.stats.vertexTexCoordBytes += vertexTexCoordBytes;

      ((std::atomic_uint32_t&)m_histograms.groupClusters[clusterCount])++;
    }

    if(m_config.useCompressedData)
    {
      compressGroup(context, groupTempStorage, groupInfo, vertexCacheLocal.data());
    }
  }

  // do actual storage & basic stats

  bool useOrderedLock = groupIndex != ~0 && context->innerThreadingActive;

  if(useOrderedLock)
  {
    // Want to enter the mutex in an ordered fashion
    // to preserve storage order from library.
    // It works without this as well, but then we don't have determinism in the
    // memory storage order of groups. And we might want to sort groups spatially
    // for more cache-efficient loading.
    while(true)
    {
      if(context->groupIndexOrdered.load() == groupIndex)
      {
        groupInfo.offsetBytes = context->groupDataOrdered.fetch_add(groupInfo.sizeBytes);

        context->groupIndexOrdered.store(groupIndex + 1);
        break;
      }
      else
      {
        std::this_thread::yield();
      }
    }
  }

  {
    std::lock_guard lock(context->groupMutex);

    geometry.bbox.lo = glm::min(groupBbox.lo, geometry.bbox.lo);
    geometry.bbox.hi = glm::max(groupBbox.hi, geometry.bbox.hi);

    if(context->lodLevel != uint32_t(group.depth))
    {
      context->lodLevel                  = uint32_t(group.depth);
      const shaderio::LodLevel* previous = group.depth ? &geometry.lodLevels[group.depth - 1] : nullptr;

      shaderio::LodLevel initLevel{};
      initLevel.clusterOffset = previous ? previous->clusterOffset + previous->clusterCount : 0;
      initLevel.groupOffset   = previous ? previous->groupOffset + previous->groupCount : 0;

      initLevel.minBoundingSphereRadius = FLT_MAX;
      initLevel.minMaxQuadricError      = FLT_MAX;

      // add new
      geometry.lodLevels.push_back(initLevel);
    }

    geometry.lodLevels[level].clusterCount += groupInfo.clusterCount;
    geometry.lodLevels[level].groupCount++;

    // USE_BLAS_SHARING
    //
    // For the BLAS sharing technique we need to figure out the conservative
    // lod range that an instance may cover. We store for each lod level
    // the smallest possible group bounding sphere as well as the smallest
    // maximum error found in any group.

    // The technique will use these values to artificially place a lod sphere
    // at the far end of an instance and evaluate its lod metric. The minima
    // ensure that there can't be any group in the instance's sphere that would
    // behave such a way that it requires lower detail.
    //
    // See `instance_classify_lod.comp.glsl` shader.

    geometry.lodLevels[level].minBoundingSphereRadius =
        std::min(geometry.lodLevels[level].minBoundingSphereRadius, group.simplified.radius);
    geometry.lodLevels[level].minMaxQuadricError =
        std::min(geometry.lodLevels[level].minMaxQuadricError, group.simplified.error);

    // stats

    geometry.clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, geometry.clusterMaxTrianglesCount);
    geometry.clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, geometry.clusterMaxVerticesCount);

    if(level == 0)
    {
      geometry.hiClustersCount += groupInfo.clusterCount;
      geometry.hiTriangleCount += groupInfo.triangleCount;
      geometry.hiVerticesCount += groupInfo.vertexCount;
    }

    geometry.totalClustersCount += groupInfo.clusterCount;
    geometry.totalTriangleCount += groupInfo.triangleCount;
    geometry.totalVerticesCount += groupInfo.vertexCount;

    // primary allocation and export
    if(useOrderedLock)
    {
      // groupInfo.offsetBytes was acquired in an orderly fashion
      if(geometry.groupData.size() < groupInfo.offsetBytes + groupInfo.sizeBytes)
      {
        geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);
      }
    }
    else
    {
      // without inner parallelism we get called linearly anyway
      groupInfo.offsetBytes = geometry.groupData.size();
      geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);

      // may also need to generate the groupIndex manually
      if(groupIndex == ~0)
      {
        groupIndex = uint32_t(geometry.groupInfos.size());
        geometry.groupInfos.resize(groupIndex + 1);
      }
    }

    geometry.groupInfos[groupIndex] = groupInfo;

    {
      GroupStorage groupStorage(&geometry.groupData[groupInfo.offsetBytes], groupInfo);

      size_t startAddress = size_t(groupStorage.group);

      // always zero, patched in during upload
      groupStorage.group->residentID        = 0;
      groupStorage.group->clusterResidentID = 0;

      // regular values
      groupStorage.group->lodLevel                             = uint8_t(level);
      groupStorage.group->stateBits                            = uint8_t(groupStateBits);
      groupStorage.group->clusterCount                         = groupInfo.clusterCount;
      groupStorage.group->traversalMetric.boundingSphereX      = group.simplified.center[0];
      groupStorage.group->traversalMetric.boundingSphereY      = group.simplified.center[1];
      groupStorage.group->traversalMetric.boundingSphereZ      = group.simplified.center[2];
      groupStorage.group->traversalMetric.boundingSphereRadius = group.simplified.radius;
      groupStorage.group->traversalMetric.maxQuadricError      = group.simplified.error;

      memcpy(groupStorage.clusters.data(), groupTempStorage.clusters.data(), groupStorage.clusters.size_bytes());

      // shared front sections
      memcpy(groupStorage.clusterGeneratingGroups.data(), groupTempStorage.clusterGeneratingGroups.data(),
             groupStorage.clusterGeneratingGroups.size_bytes());
      padZeroes(groupStorage.clusterGeneratingGroups, groupStorage.clusterBboxes.data());
      memcpy(groupStorage.clusterBboxes.data(), groupTempStorage.clusterBboxes.data(), groupStorage.clusterBboxes.size_bytes());
      memcpy(groupStorage.triangles.data(), groupTempStorage.triangles.data(), groupStorage.triangles.size_bytes());
      padZeroes(groupStorage.triangles, groupStorage.vertices.data());

      if(groupInfo.uncompressedSizeBytes)
      {
        // compressed: keep the compressed vertex blob as-is. The triangles slot stores the
        // compressed-source offset (used by decompressGroup); the positions/attributes slots store
        // the runtime (decompressed, separated) offsets so the low-detail CLAS build - which reads
        // these headers but points at the decompressed group - gets valid vertex addresses.
        size_t   frontBytes    = groupInfo.computeUncompressedSectionSize();
        size_t   posRegionByte = groupInfo.positionsByteOffset();
        uint32_t attrRunning   = 0;
        uint32_t posRunning    = 0;  // floats
        for(uint32_t c = 0; c < clusterCount; c++)
        {
          shaderio::Cluster& groupCluster = groupStorage.clusters[c];
          uint32_t           v            = uint32_t(groupCluster.vertexCountMinusOne) + 1;
          uint32_t           srcFloat     = shaderio::Cluster_getTrianglesOffset(groupCluster);
          uint32_t           srcByte = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + srcFloat);

          size_t   clusterHeaderByte = sizeof(shaderio::Group) + sizeof(shaderio::Cluster) * c;
          uint32_t attrClusterStart  = (attrRunning + 1u) & ~1u;  // floats
          uint32_t posByte           = uint32_t(posRegionByte + size_t(posRunning) * sizeof(float) - clusterHeaderByte);
          uint32_t attrByte = uint32_t(frontBytes + size_t(attrClusterStart) * sizeof(float) - clusterHeaderByte);
          shaderio::Cluster_setOffsets(groupCluster, posByte, attrByte, srcByte);

          attrRunning = attrClusterStart + clusterSeparatedAttrFloats(v, groupCluster.attributeBits);
          posRunning += 3u * v;
        }
        memcpy(groupStorage.vertices.data(), groupTempStorage.vertices.data(), groupStorage.vertices.size_bytes());
      }
      else
      {
        // uncompressed: re-pack the interleaved temp vertex data into the runtime [attributes][positions] layout.
        // The temp storage holds per-cluster [pos][nrm][tex]; positions are gathered into the trailing region.
        float*       dstV           = groupStorage.vertices.data();
        const float* srcV           = groupTempStorage.vertices.data();
        uint32_t     attrTotalFloat = groupInfo.attributesFloatCount();  // == vertexDataCount - 3*vertexCount
        uint32_t     attrRunning    = 0;
        uint32_t     posRunning     = 0;

        for(uint32_t c = 0; c < clusterCount; c++)
        {
          shaderio::Cluster& groupCluster = groupStorage.clusters[c];
          uint32_t           v            = uint32_t(groupCluster.vertexCountMinusOne) + 1;
          uint32_t           bits         = groupCluster.attributeBits;

          uint32_t tempPosFloat = shaderio::Cluster_getPositionsOffset(groupCluster);  // temp interleaved pos start
          uint32_t tempTriByte  = shaderio::Cluster_getTrianglesOffset(groupCluster);  // temp triangle byte offset

          uint32_t attrClusterStart = (attrRunning + 1u) & ~1u;  // 8-byte aligned attribute block start
          uint32_t posClusterStart  = attrTotalFloat + posRunning;

          // gather positions into the trailing region
          memcpy(dstV + posClusterStart, srcV + tempPosFloat, sizeof(float) * 3u * v);

          // copy attributes (normals/tangents then texcoords), mirroring the temp source alignment
          uint32_t srcOff = tempPosFloat + 3u * v;
          uint32_t dstOff = attrClusterStart;
          if(bits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
          {
            memcpy(dstV + dstOff, srcV + srcOff, sizeof(float) * v);
            srcOff += v;
            dstOff += v;
          }
          if(bits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
          {
            srcOff = (srcOff + 1u) & ~1u;
            dstOff = (dstOff + 1u) & ~1u;
            memcpy(dstV + dstOff, srcV + srcOff, sizeof(float) * 2u * v);
            srcOff += 2u * v;
            dstOff += 2u * v;
          }
          if(bits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1)
          {
            srcOff = (srcOff + 1u) & ~1u;
            dstOff = (dstOff + 1u) & ~1u;
            memcpy(dstV + dstOff, srcV + srcOff, sizeof(float) * 2u * v);
            srcOff += 2u * v;
            dstOff += 2u * v;
          }

          uint32_t posByte  = groupStorage.getClusterLocalOffset(c, dstV + posClusterStart);
          uint32_t attrByte = groupStorage.getClusterLocalOffset(c, dstV + attrClusterStart);
          uint32_t triByte  = groupStorage.getClusterLocalOffset(c, groupStorage.triangles.data() + tempTriByte);
          shaderio::Cluster_setOffsets(groupCluster, posByte, attrByte, triByte);

          attrRunning = dstOff;
          posRunning += 3u * v;
        }
      }
      padZeroes(groupStorage.vertices, (uint32_t*)(groupStorage.raw + groupInfo.sizeBytes));
    }
  }

  return groupIndex;
}

// Callback used by the mesoptimizer's clusterlod generator. Run once
// for each lod level (except the very last). task_count is the number
// of groups to be processed within this lod level.
// This sample only uses this callback when we intend to multi-thread within
// a single geometry. When loading scenes with many objects this is less likely
// to be used.
void Scene::clodIterationMeshoptimizer(void* intermediate_context, void* output_context, int depth, size_t task_count)
{
  TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
  GeometryStorage& geometry = context->geometry;

  context->levelGroupOffset      = geometry.groupInfos.size();
  context->levelGroupOffsetValid = true;
  geometry.groupInfos.resize(context->levelGroupOffset + task_count);


  nvutils::parallel_batches_pooled<1>(
      task_count,
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        clodBuild_iterationTask(intermediate_context, output_context, idx, threadInnerIdx);
      },
      context->processingInfo.numInnerThreads);


  context->levelGroupOffsetValid = false;
}

// callback used by mesoptimizer's clusterlod generator to provide the
// result cluster group for further processing.
int Scene::clodGroupMeshoptimizer(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, uint32_t thread_index)
{
  TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
  GeometryStorage& geometry = context->geometry;

  uint32_t groupIndex =
      context->innerThreadingActive && context->levelGroupOffsetValid ? uint32_t(context->levelGroupOffset + task_index) : ~0u;

  return context->scene.storeGroup(context, thread_index, groupIndex, group, uint32_t(cluster_count), clusters);
}

void Scene::buildGeometryLod(ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  clodConfig clodInfo = m_config.meshoptPreferRayTracing ? clodDefaultConfigRT(m_config.clusterTriangles) :
                                                           clodDefaultConfig(m_config.clusterTriangles);

  clodInfo.cluster_fill_weight  = m_config.meshoptFillWeight;
  clodInfo.cluster_split_factor = m_config.meshoptSplitFactor;
  clodInfo.max_vertices         = m_config.clusterVertices;
  clodInfo.partition_size       = m_config.clusterGroupSize;
  clodInfo.partition_spatial    = true;
  clodInfo.partition_sort       = true;

  // this only reorders triangles within cluster
  clodInfo.optimize_clusters = true;

  // sizes the cluster index pool's per-thread free lists
  clodInfo.thread_count = processingInfo.numInnerThreads;

  // account for meshopt_partitionClusters's using a target value with a higher worst case
  while((clodInfo.partition_size + clodInfo.partition_size / 3) > m_config.clusterGroupSize)
  {
    clodInfo.partition_size--;
  }

  // These control the error propagation across lod levels to
  // account for simplifying an already simplified mesh.
  clodInfo.simplify_error_merge_previous = m_config.lodErrorMergePrevious;
  clodInfo.simplify_error_merge_additive = m_config.lodErrorMergeAdditive;
  clodInfo.simplify_error_edge_limit     = m_config.lodErrorEdgeLimit;

  clodMesh inputMesh                = {};
  inputMesh.vertex_positions        = reinterpret_cast<const float*>(geometry.vertexPositions.data());
  inputMesh.vertex_count            = geometry.vertexPositions.size();
  inputMesh.vertex_positions_stride = sizeof(glm::vec3);
  inputMesh.index_count             = geometry.triangles.size() * 3;
  inputMesh.indices                 = reinterpret_cast<const uint32_t*>(geometry.triangles.data());

  float attributeWeights[9] = {};

  if(geometry.attributesWithWeights)
  {
    if(m_config.simplifyNormalWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      attributeWeights[geometry.attributeNormalOffset + 0] = m_config.simplifyNormalWeight;
      attributeWeights[geometry.attributeNormalOffset + 1] = m_config.simplifyNormalWeight;
      attributeWeights[geometry.attributeNormalOffset + 2] = m_config.simplifyNormalWeight;
    }
    if(m_config.simplifyTexCoordWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      attributeWeights[geometry.attributeTex0offset + 0] = m_config.simplifyTexCoordWeight;
      attributeWeights[geometry.attributeTex0offset + 1] = m_config.simplifyTexCoordWeight;
    }
    if(m_config.simplifyTexCoordWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
    {
      attributeWeights[geometry.attributeTex1offset + 0] = m_config.simplifyTexCoordWeight;
      attributeWeights[geometry.attributeTex1offset + 1] = m_config.simplifyTexCoordWeight;
    }
    if(m_config.simplifyTangentWeight > 0 && m_config.simplifyTangentSignWeight > 0
       && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
    {
      attributeWeights[geometry.attributeTangentOffset + 0] = m_config.simplifyTangentWeight;
      attributeWeights[geometry.attributeTangentOffset + 1] = m_config.simplifyTangentWeight;
      attributeWeights[geometry.attributeTangentOffset + 2] = m_config.simplifyTangentWeight;
      attributeWeights[geometry.attributeTangentOffset + 3] = m_config.simplifyTangentSignWeight;
    }
    if(m_config.simplifyMaterialWeight > 0 && geometry.localMaterialIDs.size() > 1)
    {
      attributeWeights[geometry.attributeMaterialOffset + 0] = m_config.simplifyMaterialWeight;
    }

    inputMesh.attribute_count          = geometry.attributesWithWeights;
    inputMesh.vertex_attributes        = geometry.vertexAttributes.data();
    inputMesh.vertex_attributes_stride = sizeof(float) * inputMesh.attribute_count;
    inputMesh.attribute_weights        = attributeWeights;
  }

  TempContext context = {processingInfo, geometry, *this};

  GroupInfo worstGroup         = {};
  worstGroup.clusterCount      = uint8_t(m_config.clusterGroupSize);
  worstGroup.vertexCount       = uint16_t(m_config.clusterGroupSize * m_config.clusterVertices);
  worstGroup.triangleCount     = uint16_t(m_config.clusterGroupSize * m_config.clusterTriangles);
  worstGroup.vertexDataCount   = worstGroup.estimateVertexDataCount(geometry.attributeBits);
  worstGroup.triangleDataCount = worstGroup.estimateTriangleDataCount(geometry.localMaterialIDs.size() > 1);
  worstGroup.sizeBytes         = worstGroup.computeSize();

  context.innerThreadingActive   = processingInfo.numInnerThreads > 1;
  context.threadGroupInfo        = worstGroup;
  context.threadGroupStorageSize = uint32_t(worstGroup.computeSize());
  context.threadGroupSize        = nvutils::align_up(context.threadGroupStorageSize, 4) + sizeof(uint32_t) * 256 * 2
                            + sizeof(uint32_t) * m_config.clusterGroupSize * m_config.clusterVertices;
  context.threadGroupDatas.resize(context.threadGroupSize * processingInfo.numInnerThreads);

  size_t reservedClusters = (geometry.triangles.size() + m_config.clusterTriangles - 1) / m_config.clusterTriangles;
  // the lod chain roughly triples the group count of the highest detail level
  size_t reservedGroups = ((reservedClusters + m_config.clusterGroupSize - 1) / m_config.clusterGroupSize) * 3;

  geometry.groupInfos.reserve(reservedGroups);
  geometry.lodLevels.reserve(32);

  clodBuild(clodInfo, inputMesh, &context, clodGroupMeshoptimizer,
            processingInfo.numInnerThreads > 1 ? clodIterationMeshoptimizer : nullptr);

  // can nuke inputs
  geometry.triangles        = {};
  geometry.vertexPositions  = {};
  geometry.vertexAttributes = {};

  // check last lod level
  geometry.lodLevelsCount            = uint32_t(geometry.lodLevels.size());
  geometry.lowDetailClusterStateBits = 0;
  if(geometry.lodLevelsCount)
  {
    shaderio::LodLevel& lastLodLevel = geometry.lodLevels.back();

    if(lastLodLevel.groupCount != 1 || lastLodLevel.clusterCount != 1)
    {
      assert(0);
      LOGE("clodBuild failed: last lod level has more than one cluster\n");
      std::exit(-1);
    }

    const GroupInfo& lastGroupInfo = geometry.groupInfos[lastLodLevel.groupOffset];
    GroupView        lastGroupView(std::span<const uint8_t>(geometry.groupData), lastGroupInfo);
    geometry.lowDetailClusterStateBits = lastGroupView.clusters.empty() ? uint8_t(0) : lastGroupView.clusters[0].stateBits;
  }

  // vectors are resized at end of lod processing,
  // but might still occupy a lot of memory
  geometry.groupInfos.shrink_to_fit();
  geometry.groupData.shrink_to_fit();
  geometry.lodLevels.shrink_to_fit();

  buildGeometryLodHierarchy(processingInfo, geometry);

  geometry.lodNodeBboxes.resize(geometry.lodNodes.size());
  computeLodBboxes_recursive(geometry, 0);

  ((std::atomic_uint32_t&)m_histograms.lodLevels[geometry.lodLevelsCount])++;

  // geometry sizes vary a lot, don't let one large one pin its high-water buffers
  // on this thread for the rest of the run
  if(size_t leaked = threadLocalArenaThreadLiveBytes())
  {
    LOGW("meshopt arena: %zu bytes still live after geometry build, trim skipped (out-of-order free?)\n", leaked);
  }
  threadLocalArenaTrimThread();
}

void Scene::buildGeometryLodHierarchy(ProcessingInfo& processingInfo, GeometryStorage& geometry)
{
  // for each lod level build hierarchy

  uint32_t lodLevelCount = geometry.lodLevelsCount;

  std::vector<Range> lodNodeRanges(lodLevelCount);

  // top root is first
  // lod-level many lod-roots next
  // then rest
  {
    uint32_t nodeOffset = 1 + lodLevelCount;

    for(uint32_t lodLevel = 0; lodLevel < lodLevelCount; lodLevel++)
    {
      const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];

      // groups as leaves
      uint32_t nodeCount = lodLevelInfo.groupCount;

      // then nodes on top
      uint32_t iterationCount = nodeCount;

      while(iterationCount > 1)
      {
        iterationCount = (iterationCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
        nodeCount += iterationCount;
      }

      // subtract root, already accounted for
      nodeCount--;

      lodNodeRanges[lodLevel].offset = nodeOffset;
      lodNodeRanges[lodLevel].count  = nodeCount;
      nodeOffset += nodeCount;
    }
    geometry.lodNodes.resize(nodeOffset);
  }

  // build per-level trees
  nvutils::parallel_batches_pooled<1>(
      lodLevelCount,
      [&](uint64_t idx, uint32_t threadInnerIdx) {
        uint32_t                  lodLevel     = uint32_t(idx);
        const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];
        const Range&              lodNodeRange = lodNodeRanges[lodLevel];

        // groups as leaves
        uint32_t nodeCount      = lodLevelInfo.groupCount;
        uint32_t nodeOffset     = lodNodeRange.offset;
        uint32_t lastNodeOffset = nodeOffset;

        for(uint32_t g = 0; g < nodeCount; g++)
        {
          uint32_t         groupID   = g + lodLevelInfo.groupOffset;
          const GroupInfo& groupInfo = geometry.groupInfos[groupID];
          GroupView        groupView(geometry.groupData, groupInfo);

          shaderio::Node& node = nodeCount == 1 ? geometry.lodNodes[1 + lodLevel] : geometry.lodNodes[nodeOffset++];

          node                                      = {};
          node.groupRange.isGroup                   = 1;
          node.groupRange.groupIndex                = groupID;
          node.groupRange.groupClusterCountMinusOne = groupInfo.clusterCount - 1;
          node.traversalMetric                      = groupView.group->traversalMetric;
        }
        // special case single node, directly stored to root section
        if(nodeCount == 1)
        {
          nodeOffset++;
        }

        // then nodes on top
        uint32_t depth          = 0;
        uint32_t iterationCount = nodeCount;

        std::vector<uint32_t>       partitionedIndices;
        std::vector<shaderio::Node> oldNodes;

        while(iterationCount > 1)
        {
          uint32_t        lastNodeCount = iterationCount;
          shaderio::Node* lastNodes     = &geometry.lodNodes[lastNodeOffset];

          // partition last nodes into children for new nodes
          partitionedIndices.resize(lastNodeCount);
          meshopt_spatialClusterPoints(partitionedIndices.data(), &lastNodes->traversalMetric.boundingSphereX,
                                       lastNodeCount, sizeof(shaderio::Node), m_config.preferredNodeWidth);

          {
            // re-order last nodes by new partition
            oldNodes.clear();
            oldNodes.insert(oldNodes.end(), lastNodes, lastNodes + lastNodeCount);

            for(uint32_t n = 0; n < lastNodeCount; n++)
            {
              lastNodes[n] = oldNodes[partitionedIndices[n]];
            }
          }

          // number of new nodes
          iterationCount = (lastNodeCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;

          // root is stored at special place
          shaderio::Node* newNodes = iterationCount == 1 ? &geometry.lodNodes[1 + lodLevel] : &geometry.lodNodes[nodeOffset];

          for(uint32_t n = 0; n < iterationCount; n++)
          {
            shaderio::Node& node          = newNodes[n];
            shaderio::Node* childrenNodes = &lastNodes[n * m_config.preferredNodeWidth];

            uint32_t childCount = std::min((n + 1) * m_config.preferredNodeWidth, lastNodeCount) - n * m_config.preferredNodeWidth;

            node                                 = {};
            node.nodeRange.isGroup               = 0;
            node.nodeRange.childCountMinusOne    = childCount - 1;
            node.nodeRange.childOffset           = lastNodeOffset + n * m_config.preferredNodeWidth;
            node.traversalMetric.maxQuadricError = 0;

            for(uint32_t c = 0; c < childCount; c++)
            {
              node.traversalMetric.maxQuadricError =
                  std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
            }

            meshopt_Bounds merged =
                meshopt_computeSphereBounds(&childrenNodes[0].traversalMetric.boundingSphereX, childCount, sizeof(shaderio::Node),
                                            &childrenNodes[0].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));

            node.traversalMetric.boundingSphereX      = merged.center[0];
            node.traversalMetric.boundingSphereY      = merged.center[1];
            node.traversalMetric.boundingSphereZ      = merged.center[2];
            node.traversalMetric.boundingSphereRadius = merged.radius;
          }

          lastNodeOffset = nodeOffset;
          nodeOffset += iterationCount;
          depth++;
        }

        nodeOffset--;
        assert(lodNodeRange.offset + lodNodeRange.count == nodeOffset);
      },
      processingInfo.numInnerThreads);

  // then setup top tree root
  {
    meshopt_Bounds merged =
        meshopt_computeSphereBounds(&geometry.lodNodes[1].traversalMetric.boundingSphereX, lodLevelCount, sizeof(shaderio::Node),
                                    &geometry.lodNodes[1].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));

    shaderio::Node& node          = geometry.lodNodes[0];
    shaderio::Node* childrenNodes = &geometry.lodNodes[1];

    node                                      = {};
    node.nodeRange.isGroup                    = 0;
    node.nodeRange.childCountMinusOne         = lodLevelCount - 1;
    node.nodeRange.childOffset                = 1;
    node.traversalMetric.boundingSphereX      = merged.center[0];
    node.traversalMetric.boundingSphereY      = merged.center[1];
    node.traversalMetric.boundingSphereZ      = merged.center[2];
    node.traversalMetric.boundingSphereRadius = merged.radius;
    node.traversalMetric.maxQuadricError      = 0;

    for(uint32_t c = 0; c < lodLevelCount; c++)
    {
      node.traversalMetric.maxQuadricError =
          std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
    }
  }
}
}  // namespace lodclusters