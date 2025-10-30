/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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