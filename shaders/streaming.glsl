/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

void streamingAgeFilter(uint residentID, uint geometryID, Group_in groupRef, bool useBlasCaching)
{
#if STREAMING_DEBUG_ADDRESSES
  if (uint64_t(groupRef) >= STREAMING_INVALID_ADDRESS_START)
  {
    streamingRW.request.errorAgeFilter = residentID;
    return;
  }
#endif

  // increase the age of a resident group  
  uint age = streaming.resident.groups.d[residentID].age;
  
  if (useBlasCaching)
  {
    uint lodLevel    = streaming.resident.groups.d[residentID].lodLevel;
    uint cachedLevel = build.geometryBuildInfos.d[geometryID].cachedLevel;
    
    // keep cached levels alive
    if (lodLevel >= cachedLevel) {
      age = 0;
    }
  }

  if (age < 0xFFFF)
  {
    age++;
    streaming.resident.groups.d[residentID].age = uint16_t(age);
  }
    
  // detect if we are over the age limit and request the group to be unloaded
  if (age > streaming.ageThreshold)
  {    
    uint unloadOffset = atomicAdd(streamingRW.request.unloadCounter, 1);
    if (unloadOffset <= streaming.request.maxUnloads) {
      streaming.request.unloadGeometryGroups.d[unloadOffset] = uvec2(geometryID, streaming.resident.groupIDs.d[residentID]);
    }
  }
}