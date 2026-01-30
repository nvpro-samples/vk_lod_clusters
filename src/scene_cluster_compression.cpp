
/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <bit>
#include <algorithm>

#include <meshoptimizer.h>

#include "scene.hpp"
#include "../shaders/attribute_encoding.h"

namespace compression {
class OutputBitStream
{
public:
  OutputBitStream() {}
  OutputBitStream(size_t byteSize, uint32_t* data) { init(byteSize, data); }

  void init(size_t byteSize, uint32_t* data)
  {
    assert(byteSize % sizeof(uint32_t) == 0);
    m_data     = data;
    m_bitsSize = byteSize * 8;
    m_bitsPos  = 0;
  }

  size_t getWrittenBitsCount() const { return m_bitsPos; }

  void write(uint32_t val, uint32_t bitCount)
  {
    assert(bitCount <= 32);
    assert(m_bitsPos + bitCount <= m_bitsSize);

    val &= bitCount == 32 ? ~0u : ((1u << bitCount) - 1);

    size_t   idxLo = m_bitsPos / 32;
    size_t   idxHi = (m_bitsPos + bitCount - 1) / 32;
    uint32_t shift = uint32_t(m_bitsPos % 32);

    if(shift == 0)
    {
      m_data[idxLo] = val;
    }
    else
    {
      m_data[idxLo] |= val << shift;
    }

    if(shift + bitCount > 32)
    {
      m_data[idxHi] = val >> (32 - shift);
    }

    m_bitsPos += bitCount;
  }

  template <typename T>
  void write(const T& tValue)
  {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    union
    {
      uint32_t u32;
      T        t;
    };

    u32 = 0;
    t   = tValue;

    write(u32, sizeof(T) * 8);
  }

private:
  uint32_t* m_data     = nullptr;
  size_t    m_bitsSize = 0;
  size_t    m_bitsPos  = 0;
};

class InputBitStream
{
public:
  InputBitStream() {}
  InputBitStream(size_t byteSize, const uint32_t* data) { init(byteSize, data); }

  void init(size_t byteSize, const uint32_t* data)
  {
    assert(byteSize % sizeof(uint32_t) == 0);
    m_data     = data;
    m_bitsPos  = 0;
    m_bitsSize = byteSize * 8;
  }

  void read(uint32_t* value, uint32_t bitCount)
  {
    assert(bitCount <= 32);
    assert(m_bitsPos + bitCount <= m_bitsSize);

    size_t   idxLo = m_bitsPos / 32;
    size_t   idxHi = (m_bitsPos + bitCount - 1) / 32;
    uint32_t shift = uint32_t(m_bitsPos % 32);

    union
    {
      uint64_t u64;
      uint32_t u32[2];
    };

    u32[0] = m_data[idxLo];
    u32[1] = m_data[idxHi];

    value[0] = uint32_t(u64 >> shift);
    value[0] &= bitCount == 32 ? ~0u : ((1u << bitCount) - 1);

    m_bitsPos += bitCount;
  }

  template <typename T>
  void read(T& value)
  {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    union
    {
      uint32_t u32;
      T        tValue;
    };
    read(&u32, sizeof(T) * 8);
    value = tValue;
  }

  size_t getBytesRead() const { return sizeof(uint32_t) * ((m_bitsPos + 31) / 32); }
  size_t getElementsRead() const { return ((m_bitsPos + 31) / 32); }


private:
  const uint32_t* m_data     = nullptr;
  size_t          m_bitsSize = 0;
  size_t          m_bitsPos  = 0;
};

template <class T, uint32_t DIM>
class ArithmeticDeCompressor
{
public:
  void init(size_t byteSize, const uint32_t* data)
  {
    m_input.init(byteSize, data);

    uint16_t outShifts;
    uint16_t outPrecs;

    m_input.read(outShifts);
    m_input.read(outPrecs);

    for(uint32_t d = 0; d < DIM; d++)
    {
      m_shifts[d]     = (outShifts >> (d * 5)) & 31;
      m_precisions[d] = ((outPrecs >> (d * 5)) & 31) + 1;
      m_input.read(m_lo[d]);
    }
  }

  size_t readVertices(size_t count, T* output, size_t strideInElements)
  {
    for(size_t v = 0; v < count; v++)
    {
      T* vec = output + v * strideInElements;
      for(uint32_t d = 0; d < DIM; d++)
      {
        uint32_t deltaBits = 0;
        m_input.read(&deltaBits, m_precisions[d]);
        vec[d] = m_lo[d] + (deltaBits << m_shifts[d]);
      }
    }

    return m_input.getBytesRead();
  }

public:
  T   m_lo[DIM];
  int m_shifts[DIM]     = {};
  int m_precisions[DIM] = {};

  InputBitStream m_input;
};

template <class T, uint32_t DIM>
class ArithmeticCompressor
{
public:
  ArithmeticCompressor()
  {
    for(uint32_t d = 0; d < DIM; d++)
    {
      m_lo[d]    = std::numeric_limits<T>::max();
      m_hi[d]    = std::numeric_limits<T>::min();
      m_masks[d] = 0;
    }
  }

  template <typename Tindices>
  void registerVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t vecStrideInElements)
  {
    m_count = count;

    for(size_t i = 0; i < count; i++)
    {
      size_t index = indices[i];
      assert(index < vecSize);
      const T* vec = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {
        m_lo[d] = std::min(m_lo[d], vec[d]);
        m_hi[d] = std::max(m_hi[d], vec[d]);
      }
    }

    for(size_t i = 0; i < count; i++)
    {
      size_t   index = indices[i];
      const T* vec   = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {
        uint32_t dv = vec[d] - m_lo[d];
        m_masks[d] |= dv;
      }
    }

    computeVertexSize();
  }

  size_t getOutputByteSize() const
  {
    // vertex bits
    size_t numDeltaBits = 0;
    for(uint32_t d = 0; d < DIM; d++)
    {
      numDeltaBits += m_precisions[d];
    }
    numDeltaBits *= m_count;

    // shift + precision + base + deltas
    return sizeof(uint32_t) * ((16 + 16 + 32 * 3 + numDeltaBits + 31) / 32);
  }

  void beginOutput(size_t byteSize, uint32_t* out)
  {
    assert(byteSize <= getOutputByteSize());

    outBits.init(byteSize, out);

    uint16_t outShifts = m_shifts[0];
    uint16_t outPrec   = m_precisions[0] - 1;

    for(uint32_t d = 1; d < DIM; d++)
    {
      outShifts |= m_shifts[d] << (d * 5);
      outPrec |= (m_precisions[d] - 1) << (d * 5);
    }

    outBits.write(outShifts);
    outBits.write(outPrec);
    for(uint32_t d = 0; d < DIM; d++)
    {
      outBits.write(m_lo[d]);
    }
  }

  template <typename Tindices>
  void outputVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t vecStrideInElements)
  {
    for(size_t i = 0; i < count; i++)
    {
      size_t index = indices[i];
      assert(index < vecSize);
      const T* vec = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {
        outBits.write((vec[d] - m_lo[d]) >> m_shifts[d], m_precisions[d]);
      }
    }
  }

public:
  T      m_lo[DIM];
  T      m_hi[DIM];
  T      m_masks[DIM];
  size_t m_count           = 0;
  int    m_shifts[DIM]     = {};
  int    m_precisions[DIM] = {};

  OutputBitStream outBits;

  void computeVertexSize()
  {
    for(uint32_t d = 0; d < DIM; ++d)
    {
      if(m_masks[d] == 0)
      {
        m_shifts[d]     = 31;
        m_precisions[d] = 1;
      }
      else
      {
        m_shifts[d] = std::countr_zero(m_masks[d]);

        const uint32_t value_range = m_hi[d] - m_lo[d];
        int            bits        = std::bit_width(value_range >> m_shifts[d]);
        m_precisions[d]            = std::max(bits, int(1));
      }
    }
  }
};
}  // namespace compression

namespace lodclusters {

void Scene::compressGroup(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo, uint32_t* vertexCacheLocal)
{
  GeometryStorage& geometry = context->geometry;

  size_t attributeStride = geometry.vertexAttributes.size() / geometry.vertexPositions.size();

  // per-cluster
  uint32_t vertexOffset     = 0;
  uint32_t vertexDataOffset = 0;
  for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
  {
    const uint32_t* localVertices = vertexCacheLocal + vertexOffset;

    shaderio::Cluster& cluster     = groupTempStorage.clusters[c];
    uint32_t           vertexCount = cluster.vertexCountMinusOne + 1;

    // will hijack indices offset for data offset storage
    cluster.indices = vertexDataOffset;

    {
      compression::ArithmeticCompressor<uint32_t, 3> compressor;

      compressor.registerVertices(vertexCount, localVertices, geometry.vertexPositions.size(),
                                  (const uint32_t*)geometry.vertexPositions.data(), 3);

      size_t compressedSize = compressor.getOutputByteSize();

      if(compressedSize >= sizeof(glm::vec3) * vertexCount)
      {
        // output uncompressed
        for(uint32_t v = 0; v < vertexCount; v++)
        {
          memcpy(&groupTempStorage.vertices[vertexDataOffset + v * 3], &geometry.vertexPositions[localVertices[v]],
                 sizeof(glm::vec3));
        }

        vertexDataOffset += 3 * vertexCount;
      }
      else
      {
        cluster.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS;
        compressor.beginOutput(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);

        compressor.outputVertices(vertexCount, localVertices, geometry.vertexPositions.size(),
                                  (const uint32_t*)geometry.vertexPositions.data(), 3);
#if 0
        {
          // validate decompressor
          compression::ArithmeticDeCompressor<uint32_t, 3> decompressor;
          decompressor.init(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);

          glm::vec3 temp[256];
          size_t    bytesRead = decompressor.readVertices(vertexCount, (uint32_t*)temp, 3);

          for(uint32_t v = 0; v < vertexCount; v++)
          {
            glm::vec3 pos = geometry.vertexPositions[localVertices[ v]];
            assert(pos.x == temp[v].x);
            assert(pos.y == temp[v].y);
            assert(pos.z == temp[v].z);
          }

          assert(bytesRead == compressedSize);
        }
#endif

        vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
      }
    }

    if(geometry.attributeNormalOffset != ~0)
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
      vertexDataOffset += vertexCount;
    }

    for(uint32_t t = 0; t < 2; t++)
    {
      shaderio::ClusterAttributeBits usedBit =
          t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      shaderio::ClusterAttributeBits compressedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 :
                                                              shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1;
      uint32_t attributeTexOffset = t == 0 ? geometry.attributeTex0offset : geometry.attributeTex1offset;

      if(geometry.attributeBits & usedBit)
      {
        compression::ArithmeticCompressor<uint32_t, 2> compressor;

        compressor.registerVertices(vertexCount, localVertices, geometry.vertexPositions.size(),
                                    (const uint32_t*)(geometry.vertexAttributes.data() + attributeTexOffset), attributeStride);
        size_t compressedSize = compressor.getOutputByteSize();

        if(compressedSize >= sizeof(glm::vec2) * vertexCount)
        {
          // output uncompressed
          for(uint32_t v = 0; v < vertexCount; v++)
          {
            const glm::vec2* attribute =
                (const glm::vec2*)&geometry.vertexAttributes[localVertices[v] * attributeStride + attributeTexOffset];

            memcpy(&groupTempStorage.vertices[vertexDataOffset + v * 2], attribute, sizeof(glm::vec2));
          }

          vertexDataOffset += 2 * vertexCount;
        }
        else
        {
          cluster.attributeBits |= compressedBit;
          compressor.beginOutput(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);

          compressor.outputVertices(vertexCount, localVertices, geometry.vertexPositions.size(),
                                    (const uint32_t*)(geometry.vertexAttributes.data() + attributeTexOffset), attributeStride);

          vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
        }
      }
    }


    vertexOffset += vertexCount;
  }

  context->processingInfo.stats.vertexCompressedBytes += sizeof(uint32_t) * vertexDataOffset;

  groupInfo.uncompressedSizeBytes       = groupInfo.sizeBytes;
  groupInfo.uncompressedVertexDataCount = groupInfo.vertexDataCount;
  groupInfo.vertexDataCount             = vertexDataOffset;
  groupInfo.sizeBytes                   = groupInfo.computeSize();
}


void Scene::decompressGroup(const GroupInfo& info, const GroupView& groupSrc, void* dstWriteOnly, size_t dstSize)
{
  // assume write-only destination (uncached write-combined memory)

  GroupInfo uncompressedInfo       = info;
  uncompressedInfo.sizeBytes       = info.uncompressedSizeBytes;
  uncompressedInfo.vertexDataCount = info.uncompressedVertexDataCount;

  GroupStorage groupDstWriteOnly(dstWriteOnly, uncompressedInfo);
  memcpy(dstWriteOnly, groupSrc.raw, info.computeUncompressedSectionSize());

  uint32_t indicesOffset = 0;
  for(uint32_t c = 0; c < info.clusterCount; c++)
  {

    shaderio::Cluster&       clusterDstWriteOnly = groupDstWriteOnly.clusters[c];
    const shaderio::Cluster& clusterSrc          = groupSrc.clusters[c];
    uint32_t                 triangleCount       = clusterSrc.triangleCountMinusOne + 1;
    uint32_t                 vertexCount         = clusterSrc.vertexCountMinusOne + 1;
    uint32_t                 vertexDataOffset    = clusterSrc.vertices;

    // get pointers, start at cluster destination vertex data
    uint32_t* dstData = groupDstWriteOnly.getClusterLocalData(c, clusterSrc.vertices);
    // `cluster.indices` actually stores the location to the compressed vertex data
    const uint32_t* srcData = (const uint32_t*)groupSrc.getClusterIndices(c);

    // correct indices offset
    clusterDstWriteOnly.indices = groupDstWriteOnly.getClusterLocalOffset(c, groupDstWriteOnly.indices.data() + indicesOffset);
    indicesOffset += triangleCount * 3;

    uint32_t dstOffset = 0;

    // positions
    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS)
    {
      ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
      assert(srcSize >= 0);

      compression::ArithmeticDeCompressor<uint32_t, 3> decompressor;
      decompressor.init(size_t(srcSize), srcData);
      srcData += decompressor.readVertices(vertexCount, dstData + dstOffset, 3) / sizeof(uint32_t);
      dstOffset += 3 * vertexCount;
    }
    else
    {
      memcpy(dstData, srcData, sizeof(glm::vec3) * vertexCount);
      srcData += 3 * vertexCount;
      dstOffset += 3 * vertexCount;
    }

    // normals
    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {
      memcpy(dstData + dstOffset, srcData, sizeof(uint32_t) * vertexCount);
      srcData += vertexCount;
      dstOffset += vertexCount;
    }


    for(uint32_t t = 0; t < 2; t++)
    {
      shaderio::ClusterAttributeBits usedBit =
          t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      shaderio::ClusterAttributeBits compressedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 :
                                                              shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1;

      // texcoords
      if((clusterSrc.attributeBits & (usedBit | compressedBit)) == (usedBit | compressedBit))
      {
        ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
        assert(srcSize >= 0);

        compression::ArithmeticDeCompressor<uint32_t, 2> decompressor;
        decompressor.init(size_t(srcSize), srcData);
        srcData += decompressor.readVertices(vertexCount, dstData + dstOffset, 2) / sizeof(uint32_t);
        dstOffset += 2 * vertexCount;
      }
      else if(clusterSrc.attributeBits & usedBit)
      {
        // align
        dstOffset = (dstOffset + 1) & ~1;

        memcpy(dstData + dstOffset, srcData, sizeof(glm::vec2) * vertexCount);

        srcData += 2 * vertexCount;
        dstOffset += 2 * vertexCount;
      }
    }


    assert(size_t(dstData + dstOffset) <= size_t(dstWriteOnly) + dstSize);
  }
}

}  // namespace lodclusters