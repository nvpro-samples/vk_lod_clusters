/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Renders the memory state of the persistent clas allocator for the UI.
  One image row covers a whole number of allocator sectors, one pixel a
  fixed number of allocation units.

  Compiled three times, selected by `STREAM_ALLOCATOR_VIS_PASS`:

  - `..._PASS_MEMORY`:    a thread is one pixel, popcounts `usedBits` over the
                          units it covers, giving an occupancy heat map.
  - `..._PASS_GROUPS`:    a thread is one resident group and paints its allocation
                          on top. Only pixels fully covered are painted, so
                          neighboring allocations stay separated.
  - `..._PASS_HISTOGRAM`: a thread is one resident group and bins its allocation
                          size for the UI to read back.

  The persistently resident lowest detail clas live outside the allocator's
  memory (see `m_clasLowDetailBuffer`), so only active groups are covered.
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"

////////////////////////////////////////////

layout(push_constant) uniform pushData
{
  StreamingAllocatorVisConstants push;
};

layout(binding = BINDINGS_ALLOCATOR_VIS_IMAGE, set = 0, rgba8) uniform image2D visImage;

layout(local_size_x = STREAM_ALLOCATOR_VIS_WORKGROUP) in;

////////////////////////////////////////////

#define COLOR_SEPARATOR vec3(0.04, 0.04, 0.05)
#define COLOR_FREE vec3(0.16, 0.17, 0.21)
// dimmed so the group colors on top stand out
#define COLOR_USED vec3(0.42, 0.44, 0.48)
// allocation padding, the difference between allocated and actual clas size
#define COLOR_WASTED vec3(0.85, 0.12, 0.06)

uint histogramBin(uint sizeInUnits)
{
  uint bin = ((sizeInUnits - 1) * STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS) / max(1u, push.maxAllocationSize);
  return min(bin, uint(STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS - 1));
}

vec3 colorizeGroup(uint residentID)
{
  return vec3(unpackUnorm4x8(murmurHash(residentID ^ push.colorXor)).xyz * 0.5 + 0.3);
}

////////////////////////////////////////////

#if STREAM_ALLOCATOR_VIS_PASS == STREAM_ALLOCATOR_VIS_PASS_HISTOGRAM

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);

  SceneStreaming_in streamingRef = SceneStreaming_in(push.streamingAddress);

  // the host count can lag the device by a frame, take the smaller of the two
  if(threadID >= push.groupCount || threadID >= streamingRef.d.resident.activeGroupsCount)
    return;

  uint residentID = streamingRef.d.resident.activeGroups.d[threadID];
  uint allocSize  = streamingRef.d.resident.groupClasSizes.d[residentID].x;

  if(allocSize != 0)
  {
    atomicAdd(uint32s_inout(push.histogramAddress).d[histogramBin(allocSize)], 1);
  }
}

#elif STREAM_ALLOCATOR_VIS_PASS == STREAM_ALLOCATOR_VIS_PASS_MEMORY

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);

  // dispatched over the mapped rectangle only
  uint x = threadID % push.usedWidth;
  uint y = threadID / push.usedWidth;

  if(y >= push.usedHeight)
    return;

  uint row    = y / push.rowPitch;
  uint rowSub = y - (row * push.rowPitch);

  vec3 color = COLOR_SEPARATOR;

  if(rowSub < push.rowContent)
  {
    uint unitBegin = row * push.unitsPerRow + x * push.unitsPerPixel;
    uint unitEnd   = min(unitBegin + push.unitsPerPixel, push.totalUnits);

    if(unitBegin < unitEnd)
    {
      uint32s_inout usedBits = SceneStreaming_in(push.streamingAddress).d.clasAllocator.usedBits;

      uint firstWord = unitBegin >> 5;
      uint lastWord  = (unitEnd - 1) >> 5;
      uint usedCount = 0;

      for(uint w = firstWord; w <= lastWord; w++)
      {
        uint bits = usedBits.d[w];
        if(w == firstWord)
        {
          bits &= ~0u << (unitBegin & 31);
        }
        if(w == lastWord && (unitEnd & 31) != 0)
        {
          bits &= ~(~0u << (unitEnd & 31));
        }
        usedCount += bitCount(bits);
      }

      color = mix(COLOR_FREE, COLOR_USED, float(usedCount) / float(unitEnd - unitBegin));
    }
  }

  imageStore(visImage, ivec2(x, y), vec4(color, 1.0));
}

#else  // STREAM_ALLOCATOR_VIS_PASS_GROUPS

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);

  SceneStreaming_in streamingRef = SceneStreaming_in(push.streamingAddress);

  // the host count can lag the device by a frame, take the smaller of the two
  if(threadID >= push.groupCount || threadID >= streamingRef.d.resident.activeGroupsCount)
    return;

  uint     residentID = streamingRef.d.resident.activeGroups.d[threadID];
  Group_in groupRef   = streamingRef.d.resident.groups.d[residentID].group;

  // same as the unload kernel does, all clas of a group are allocated together
  uint64_t firstClasAddress = streamingRef.d.resident.clasAddresses.d[groupRef.d.clusterResidentID];
  uint64_t firstClasOffset  = firstClasAddress - streamingRef.d.resident.clasBaseAddress;

  uint granularityByteShift = streamingRef.d.clasAllocator.granularityByteShift;
  uint allocPos             = uint(firstClasOffset >> granularityByteShift);

  // allocation size is in units, wasted size in bytes
  uvec2 groupSize      = streamingRef.d.resident.groupClasSizes.d[residentID];
  uint  allocSize      = groupSize.x;
  uint  wastedByteSize = groupSize.y;

  if(allocSize == 0 || allocPos + allocSize > push.totalUnits)
    return;

  uint allocByteSize = allocSize << granularityByteShift;
  uint dataByteSize  = allocByteSize - min(wastedByteSize, allocByteSize);
  uint dataSize      = (dataByteSize + ((1u << granularityByteShift) - 1)) >> granularityByteShift;

  uint row = allocPos / push.unitsPerRow;
  if(row * push.rowPitch >= push.usedHeight)
    return;

  uint localBegin = allocPos - (row * push.unitsPerRow);

  // round the start up and the end down, so only fully covered pixels are painted
  uint colBegin     = (localBegin + push.unitsPerPixel - 1) / push.unitsPerPixel;
  uint colEnd       = min((localBegin + allocSize) / push.unitsPerPixel, push.usedWidth);
  uint colDataEnd   = (localBegin + dataSize) / push.unitsPerPixel;
  vec3 groupColor   = colorizeGroup(residentID);
  vec3 wastedColor  = mix(COLOR_WASTED, groupColor, 0.25);

  for(uint x = colBegin; x < colEnd; x++)
  {
    vec4 color = vec4(x < colDataEnd ? groupColor : wastedColor, 1.0);
    for(uint sub = 0; sub < push.rowContent; sub++)
    {
      imageStore(visImage, ivec2(x, row * push.rowPitch + sub), color);
    }
  }
}

#endif
