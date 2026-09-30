
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    return sizeof(uint32_t) * ((16 + 16 + 32 * DIM + numDeltaBits + 31) / 32);
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

// Per cluster the triangle region holds `triangleCount * 3` local index bytes, optionally followed
// by `triangleCount` per-triangle material bytes. The indices go through meshoptimizer's meshlet
// codec; the material bytes get a per-cluster palette, as a cluster typically mixes only two or
// three distinct values even when its geometry has many material slots.
//
// Compressed layout, per cluster in order, each block starting 4-byte aligned:
//   uint32 encodedSize     ( 0 means the indices could not be shrunk and follow raw )
//   encodedSize bytes      ( or triangleCount * 3 raw index bytes )
// and when the cluster has per-triangle materials:
//   uint8  paletteCount    ( 0 means the palette did not pay off and raw bytes follow )
//   uint8  palette[paletteCount]
//   bits   one palette index per triangle, ceil(log2(paletteCount)) bits each, none for a
//          single entry cluster
//
// Nothing needs a per-cluster offset: both sides walk the clusters in order.
//
// The codec is lossless except that it may cyclically rotate the corners of a triangle. Winding
// and therefore geometry are preserved, but the provoking vertex can change. Nothing here depends
// on it: per-triangle materials are stored per triangle rather than per corner, facet normals come
// from the cross product, and the shaders interpolate from all three corners. Ray tracing does
// resolve a handful of silhouette pixels differently, as the intersection math is not bit-wise
// invariant under corner order.
static inline uint32_t compressedTriangleBlockAlign(uint32_t offset)
{
  return (offset + 3u) & ~3u;
}

// bits needed to index a palette of `count` entries, 0 when there is nothing to choose
static inline uint32_t paletteIndexBits(uint32_t count)
{
  return count <= 1 ? 0 : uint32_t(std::bit_width(count - 1));
}

// Writes the palette form of a cluster's material bytes, or the raw bytes when that is smaller.
// Returns the number of bytes written.
static size_t encodeClusterMaterials(const uint8_t* materials, uint32_t triangleCount, uint8_t* dst)
{
  uint8_t  palette[256];
  uint8_t  valueToIndex[256];
  bool     seen[256]    = {};
  uint32_t paletteCount = 0;

  for(uint32_t t = 0; t < triangleCount; t++)
  {
    uint8_t value = materials[t];
    if(!seen[value])
    {
      seen[value]             = true;
      valueToIndex[value]     = uint8_t(paletteCount);
      palette[paletteCount++] = value;
    }
  }

  uint32_t indexBits  = paletteIndexBits(paletteCount);
  size_t   indexBytes = (size_t(triangleCount) * indexBits + 7) / 8;

  // both forms carry the count byte, so only compare what follows it
  if(paletteCount + indexBytes >= triangleCount)
  {
    dst[0] = 0;
    memcpy(dst + 1, materials, triangleCount);
    return 1 + triangleCount;
  }

  dst[0] = uint8_t(paletteCount);
  memcpy(dst + 1, palette, paletteCount);

  uint8_t* bits = dst + 1 + paletteCount;
  memset(bits, 0, indexBytes);

  for(uint32_t t = 0; t < triangleCount; t++)
  {
    uint32_t index   = valueToIndex[materials[t]];
    size_t   bitPos  = size_t(t) * indexBits;
    uint32_t bitFree = 8 - uint32_t(bitPos % 8);

    bits[bitPos / 8] |= uint8_t(index << (bitPos % 8));
    if(indexBits > bitFree)
    {
      bits[bitPos / 8 + 1] |= uint8_t(index >> bitFree);
    }
  }

  return 1 + paletteCount + indexBytes;
}

// Reverses `encodeClusterMaterials`, returns the number of source bytes consumed.
static size_t decodeClusterMaterials(const uint8_t* src, uint32_t triangleCount, uint8_t* dst)
{
  uint32_t paletteCount = src[0];

  if(paletteCount == 0)
  {
    memcpy(dst, src + 1, triangleCount);
    return 1 + triangleCount;
  }

  const uint8_t* palette   = src + 1;
  uint32_t       indexBits = paletteIndexBits(paletteCount);

  if(indexBits == 0)
  {
    memset(dst, palette[0], triangleCount);
    return 1 + paletteCount;
  }

  const uint8_t* bits = palette + paletteCount;
  uint32_t       mask = (1u << indexBits) - 1;

  for(uint32_t t = 0; t < triangleCount; t++)
  {
    size_t   bitPos  = size_t(t) * indexBits;
    uint32_t bitFree = 8 - uint32_t(bitPos % 8);
    uint32_t index   = uint32_t(bits[bitPos / 8]) >> (bitPos % 8);

    if(indexBits > bitFree)
    {
      index |= uint32_t(bits[bitPos / 8 + 1]) << bitFree;
    }

    dst[t] = palette[index & mask];
  }

  return 1 + paletteCount + (size_t(triangleCount) * indexBits + 7) / 8;
}

void Scene::compressGroupTriangles(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo)
{
  std::vector<uint8_t>& scratch = context->tempTriangleData;
  if(scratch.size() < groupInfo.triangleDataCount + size_t(groupInfo.clusterCount) * (sizeof(uint32_t) + 3))
  {
    scratch.resize(groupInfo.triangleDataCount + size_t(groupInfo.clusterCount) * (sizeof(uint32_t) + 3) + 64);
  }

  // worst case output of a single encode, the codec may exceed the raw size
  uint8_t encoded[SHADERIO_MAX_CLUSTER_TRIANGLES * 4 + 64];

  const uint8_t* src    = groupTempStorage.triangles.data();
  uint32_t       dstPos = 0;

  for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
  {
    const shaderio::Cluster& cluster       = groupTempStorage.clusters[c];
    uint32_t                 triangleCount = uint32_t(cluster.triangleCountMinusOne) + 1;
    uint32_t                 rawSize       = triangleCount * 3;

    size_t encodedSize = meshopt_encodeMeshlet(encoded, sizeof(encoded), nullptr, 0, src, triangleCount);

    dstPos = compressedTriangleBlockAlign(dstPos);

    // a failed or unprofitable encode falls back to the raw indices
    bool     useEncoded = encodedSize != 0 && encodedSize < rawSize;
    uint32_t header     = useEncoded ? uint32_t(encodedSize) : 0;

    memcpy(&scratch[dstPos], &header, sizeof(header));
    dstPos += uint32_t(sizeof(header));

    if(useEncoded)
    {
#if 0
      {
        // validate decoder, the codec is lossless except that it may cyclically rotate
        // the corners of a triangle (same winding, different provoking vertex)
        alignas(16) uint8_t back[SHADERIO_MAX_CLUSTER_TRIANGLES * 3 + 16];
        assert(meshopt_decodeMeshlet(nullptr, 0, 4, back, triangleCount, 3, encoded, encodedSize) == 0);

        for(uint32_t t = 0; t < triangleCount; t++)
        {
          const uint8_t* a = src + t * 3;
          const uint8_t* b = back + t * 3;
          assert((b[0] == a[0] && b[1] == a[1] && b[2] == a[2]) || (b[0] == a[1] && b[1] == a[2] && b[2] == a[0])
                 || (b[0] == a[2] && b[1] == a[0] && b[2] == a[1]));
        }
      }
#endif
      memcpy(&scratch[dstPos], encoded, encodedSize);
      dstPos += uint32_t(encodedSize);
    }
    else
    {
      memcpy(&scratch[dstPos], src, rawSize);
      dstPos += rawSize;
    }
    src += rawSize;

    if(cluster.localMaterialID == SHADERIO_PER_TRIANGLE_MATERIALS)
    {
      dstPos += uint32_t(encodeClusterMaterials(src, triangleCount, &scratch[dstPos]));
      src += triangleCount;
    }
  }

  assert(size_t(src - groupTempStorage.triangles.data()) == groupInfo.triangleDataCount);

  // only adopt the compressed form when it actually is smaller, otherwise the group would
  // grow and `triangleDataCount` could overflow its bitfield
  if(dstPos >= groupInfo.triangleDataCount)
  {
    return;
  }

  memcpy(groupTempStorage.triangles.data(), scratch.data(), dstPos);

  context->processingInfo.stats.triangleCompressedBytes += dstPos;

  groupInfo.uncompressedTriangleDataCount = groupInfo.triangleDataCount;
  groupInfo.triangleDataCount             = dstPos;
}

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

    // hijack the triangles offset slot to store the compressed vertex-data offset
    // (positions slot keeps the uncompressed destination offset set by the LOD builder)
    shaderio::Cluster_setOffsets(cluster, shaderio::Cluster_getPositionsOffset(cluster), 0, vertexDataOffset);

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

  // capture the uncompressed total before any of the counts are replaced
  groupInfo.uncompressedSizeBytes       = groupInfo.sizeBytes;
  groupInfo.uncompressedVertexDataCount = groupInfo.vertexDataCount;
  groupInfo.vertexDataCount             = vertexDataOffset;

  compressGroupTriangles(context, groupTempStorage, groupInfo);

  groupInfo.sizeBytes = groupInfo.computeSize();
}


// Per-triangle material bytes live inside the group's triangle region, whose layout differs
// between the raw and the compressed form, and in the compressed form they are palette encoded.
// Callers must go through this rather than indexing that region themselves.
// `triangleMaterials` receives the decoded bytes of all clusters back to back and must hold
// `info.triangleCount` of them, `perCluster` points into it, null for clusters without.
void Scene::getGroupTriangleMaterials(const GroupInfo& info, const GroupView& groupView, uint8_t* triangleMaterials, const uint8_t** perCluster)
{
  const uint8_t* src        = groupView.triangles.data();
  const bool     compressed = info.uncompressedTriangleDataCount != 0;
  uint32_t       pos        = 0;
  uint32_t       dstPos     = 0;

  for(uint32_t c = 0; c < info.clusterCount; c++)
  {
    const shaderio::Cluster& cluster       = groupView.clusters[c];
    uint32_t                 triangleCount = uint32_t(cluster.triangleCountMinusOne) + 1;

    if(compressed)
    {
      pos = compressedTriangleBlockAlign(pos);

      uint32_t encodedSize;
      memcpy(&encodedSize, src + pos, sizeof(encodedSize));
      pos += uint32_t(sizeof(encodedSize));
      pos += encodedSize ? encodedSize : triangleCount * 3;
    }
    else
    {
      pos += triangleCount * 3;
    }

    if(cluster.localMaterialID != SHADERIO_PER_TRIANGLE_MATERIALS)
    {
      perCluster[c] = nullptr;
      continue;
    }

    perCluster[c] = triangleMaterials + dstPos;

    if(compressed)
    {
      pos += uint32_t(decodeClusterMaterials(src + pos, triangleCount, triangleMaterials + dstPos));
    }
    else
    {
      memcpy(triangleMaterials + dstPos, src + pos, triangleCount);
      pos += triangleCount;
    }
    dstPos += triangleCount;
  }

  assert(pos <= info.triangleDataCount);
  assert(dstPos <= info.triangleCount);
}

// Reverses `compressGroupTriangles`, see the layout description there.
void Scene::decompressGroupTriangles(const GroupInfo& info, const GroupView& groupSrc, GroupStorage& groupDst)
{
  // meshopt_decodeMeshlet writes in 4-byte units, so it needs `align(triangleCount * 3, 4)` bytes
  // of aligned space. The per-cluster destinations inside the group are not aligned, decode into
  // this and copy the exact bytes over. It is a cached scratch copy, so the cost is negligible.
  alignas(16) uint8_t decoded[SHADERIO_MAX_CLUSTER_TRIANGLES * 3 + 16];

  const uint8_t* src    = groupSrc.triangles.data();
  uint8_t*       dst    = groupDst.triangles.data();
  uint32_t       srcPos = 0;
  uint32_t       dstPos = 0;

  for(uint32_t c = 0; c < info.clusterCount; c++)
  {
    const shaderio::Cluster& cluster       = groupSrc.clusters[c];
    uint32_t                 triangleCount = uint32_t(cluster.triangleCountMinusOne) + 1;
    uint32_t                 rawSize       = triangleCount * 3;

    srcPos = compressedTriangleBlockAlign(srcPos);

    uint32_t encodedSize;
    memcpy(&encodedSize, src + srcPos, sizeof(encodedSize));
    srcPos += uint32_t(sizeof(encodedSize));

    if(encodedSize)
    {
      [[maybe_unused]] int result = meshopt_decodeMeshlet(nullptr, 0, 4, decoded, triangleCount, 3, src + srcPos, encodedSize);
      assert(result == 0 && "meshlet triangle decode failed");

      memcpy(dst + dstPos, decoded, rawSize);
      srcPos += encodedSize;
    }
    else
    {
      memcpy(dst + dstPos, src + srcPos, rawSize);
      srcPos += rawSize;
    }
    dstPos += rawSize;

    if(cluster.localMaterialID == SHADERIO_PER_TRIANGLE_MATERIALS)
    {
      srcPos += uint32_t(decodeClusterMaterials(src + srcPos, triangleCount, dst + dstPos));
      dstPos += triangleCount;
    }
  }

  assert(srcPos <= info.triangleDataCount);
  assert(dstPos == info.uncompressedTriangleDataCount);
}

void Scene::decompressGroup(const GroupInfo& info, const GroupView& groupSrc, void* dstWriteOnly, size_t dstSize, std::vector<uint32_t>& scratch)
{
  // The destination is write-combined (uncached) staging memory: reads are extremely slow and only
  // strictly sequential writes combine into full cache-line bursts. Decode the whole group into a
  // cached scratch buffer (ordinary reads/writes, any order), then flush it to the destination in a
  // single sequential memcpy.

  // GroupStorage aligns its sub-sections off the absolute base address, while
  // computeSize/computeRuntimeVerticesOffset size the blob assuming a 16-aligned base; both the
  // scratch and the destination must be 16-byte aligned so their layouts match byte-for-byte.
  assert((reinterpret_cast<size_t>(dstWriteOnly) & 15) == 0 && "group blob destination must be 16-byte aligned");

  GroupInfo uncompressedInfo         = info;
  uncompressedInfo.sizeBytes         = info.uncompressedSizeBytes;
  uncompressedInfo.vertexDataCount   = info.uncompressedVertexDataCount;
  uncompressedInfo.triangleDataCount = info.getRuntimeTriangleDataCount();

  const size_t usedBytes = uncompressedInfo.positionsByteOffset() + uncompressedInfo.positionsByteSize();
  assert(usedBytes <= dstSize);

  if(scratch.size() * sizeof(uint32_t) < usedBytes + 16)
    scratch.resize(usedBytes / sizeof(uint32_t) + 8);
  void* dst = reinterpret_cast<void*>(nvutils::align_up(size_t(scratch.data()), 16));

  GroupStorage groupDst(dst, uncompressedInfo);

  // everything ahead of the triangle region is stored as is
  memcpy(dst, groupSrc.raw, info.computeUncompressedSectionSize());

  if(info.uncompressedTriangleDataCount)
  {
    decompressGroupTriangles(info, groupSrc, groupDst);
  }
  else
  {
    memcpy(groupDst.triangles.data(), groupSrc.triangles.data(), info.triangleDataCount);
  }

  // the scratch is reused across groups, so zero the alignment gap that used to come along
  // with the copy of the whole front section
  {
    uint8_t* trianglesEnd = groupDst.triangles.data() + uncompressedInfo.triangleDataCount;
    memset(trianglesEnd, 0, size_t(groupDst.vertices.data()) - size_t(trianglesEnd));
  }

  // runtime layout is [attributes][positions]; positions are gathered into the trailing region
  uint32_t  attrTotalFloat      = uncompressedInfo.attributesFloatCount();
  uint32_t  attrRunning         = 0;
  uint32_t  posRunning          = 0;
  uint32_t  trianglesDataOffset = 0;
  uint32_t* dstVerts            = (uint32_t*)groupDst.vertices.data();
  for(uint32_t c = 0; c < info.clusterCount; c++)
  {

    shaderio::Cluster&       clusterDst    = groupDst.clusters[c];
    const shaderio::Cluster& clusterSrc    = groupSrc.clusters[c];
    uint32_t                 triangleCount = clusterSrc.triangleCountMinusOne + 1;
    uint32_t                 vertexCount   = clusterSrc.vertexCountMinusOne + 1;

    // the triangles slot of the compressed cluster stores the location of the compressed vertex data
    const uint32_t* srcData = (const uint32_t*)groupSrc.getClusterIndices(c);

    // separate runtime destinations for this cluster
    uint32_t  attrClusterStart = (attrRunning + 1u) & ~1u;  // 8-byte aligned attribute block start
    uint32_t  posClusterStart  = attrTotalFloat + posRunning;
    uint32_t* dstPos           = dstVerts + posClusterStart;
    uint32_t* dstAttr          = dstVerts + attrClusterStart;

    // set the runtime cluster offsets (positions region, attributes region, real triangle indices)
    uint32_t posByte  = groupDst.getClusterLocalOffset(c, dstPos);
    uint32_t attrByte = groupDst.getClusterLocalOffset(c, dstAttr);
    uint32_t triByte  = groupDst.getClusterLocalOffset(c, groupDst.triangles.data() + trianglesDataOffset);
    shaderio::Cluster_setOffsets(clusterDst, posByte, attrByte, triByte);
    trianglesDataOffset += triangleCount * (clusterSrc.localMaterialID == SHADERIO_PER_TRIANGLE_MATERIALS ? 4 : 3);

    // positions -> trailing positions region (read from compressed source in source order)
    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS)
    {
      ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
      assert(srcSize >= 0);

      compression::ArithmeticDeCompressor<uint32_t, 3> decompressor;
      decompressor.init(size_t(srcSize), srcData);
      srcData += decompressor.readVertices(vertexCount, dstPos, 3) / sizeof(uint32_t);
    }
    else
    {
      memcpy(dstPos, srcData, sizeof(glm::vec3) * vertexCount);
      srcData += 3 * vertexCount;
    }

    // attributes -> attributes region
    uint32_t attrOff = 0;

    // normals
    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {
      memcpy(dstAttr + attrOff, srcData, sizeof(uint32_t) * vertexCount);
      srcData += vertexCount;
      attrOff += vertexCount;
    }

    for(uint32_t t = 0; t < 2; t++)
    {
      shaderio::ClusterAttributeBits usedBit =
          t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      shaderio::ClusterAttributeBits compressedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 :
                                                              shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1;

      // texcoords, 8-byte (2-float) aligned within the attributes block to match the shader accessor
      if((clusterSrc.attributeBits & (usedBit | compressedBit)) == (usedBit | compressedBit))
      {
        attrOff = (attrOff + 1) & ~1;

        ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
        assert(srcSize >= 0);

        compression::ArithmeticDeCompressor<uint32_t, 2> decompressor;
        decompressor.init(size_t(srcSize), srcData);
        srcData += decompressor.readVertices(vertexCount, dstAttr + attrOff, 2) / sizeof(uint32_t);
        attrOff += 2 * vertexCount;
      }
      else if(clusterSrc.attributeBits & usedBit)
      {
        attrOff = (attrOff + 1) & ~1;

        memcpy(dstAttr + attrOff, srcData, sizeof(glm::vec2) * vertexCount);

        srcData += 2 * vertexCount;
        attrOff += 2 * vertexCount;
      }
    }

    attrRunning = attrClusterStart + attrOff;
    posRunning += 3 * vertexCount;

    assert(size_t(dstPos + 3 * vertexCount) <= size_t(dst) + usedBytes);
    assert(size_t(dstAttr + attrOff) <= size_t(dst) + usedBytes);
  }

  // single sequential flush to the write-combined destination
  memcpy(dstWriteOnly, dst, usedBytes);
}


}  // namespace lodclusters