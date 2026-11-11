/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef STREAMING_GLSL
#define STREAMING_GLSL

#ifndef USE_HIT_STREAMING_LOD
#define USE_HIT_STREAMING_LOD 0
#endif

void streamingRegisterTraversed(uint residentID)
{
  streaming.resident.groups.d[residentID].age = uint16_t(0);
}

//////////////////////////////////////////////////////////////////////////
// hit tracking helpers, see shaderio_streaming.h for the packed layout of
// the `groupHitFrames` entries

// this frame's stamp, see STREAMING_HITFRAME_FRAME_BASE
uint streamingHitCyclicFrame()
{
  return STREAMING_HITFRAME_CYCLIC(streaming.frameIndex);
}

// age of a hit entry in frames, STREAMING_HITFRAME_FRAME_MASK for never
uint streamingHitFrameDelta(uint hitValue)
{
  uint stamp = (hitValue >> STREAMING_HITFRAME_FRAME_SHIFT) & STREAMING_HITFRAME_FRAME_MASK;
  uint now   = streamingHitCyclicFrame();
  if (stamp == 0)
    return STREAMING_HITFRAME_FRAME_MASK;
  // above `now` is the previous period
  return stamp <= now ? now - stamp : now + STREAMING_HITFRAME_PERIOD - stamp;
}

// stamp of an entry `delta` frames old, keeps the atomicMax order across the wrap
uint streamingHitStamp(uint delta)
{
  return delta < STREAMING_HITFRAME_FRAME_BASE ? streamingHitCyclicFrame() - delta : 0;
}

// the width is stored inverted so that a single atomicMax yields
// "newest frame wins, smallest width wins within a frame"
uint streamingEncodeHitWidth(float widthObjectSpace)
{
  float enc = (log2(max(widthObjectSpace, 1e-30f)) - STREAMING_HITLOD_WIDTH_LOG2_MIN) * STREAMING_HITLOD_WIDTH_LOG2_SCALE;
  return STREAMING_HITFRAME_WIDTH_MASK - uint(clamp(enc, 0.0f, float(STREAMING_HITFRAME_WIDTH_MASK)));
}

float streamingDecodeHitWidth(uint hitValue)
{
  uint enc = STREAMING_HITFRAME_WIDTH_MASK - (hitValue & STREAMING_HITFRAME_WIDTH_MASK);
  return exp2(float(enc) / STREAMING_HITLOD_WIDTH_LOG2_SCALE + STREAMING_HITLOD_WIDTH_LOG2_MIN);
}

// hit registration value for the current frame. Always sets the no-finer bit,
// see STREAMING_HITFRAME_NO_FINER_BIT for why this is safe.
uint streamingMakeHitValue(uint invWidth)
{
  return STREAMING_HITFRAME_NO_FINER_BIT | (streamingHitCyclicFrame() << STREAMING_HITFRAME_FRAME_SHIFT) | invWidth;
}

void streamingAgeFilter(uint residentID, uint geometryID, Group_in groupRef, bool useBlasCaching, bool useHitStreaming)
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
  
  bool keep = false;
  if (useBlasCaching)
  {
    uint lodLevel    = streaming.resident.groups.d[residentID].lodLevel;
    // the cycle minimum, not this frame's value: instances in a skipped partition
    // contribute nothing to `cachedLevel` but still reference their cached blas
    uint cachedLevel = build.geometryCachedRetain.d[geometryID];
    
    // keep cached levels alive
    if (lodLevel >= cachedLevel) {
      age = 0;
      keep = true;
    }
  }

  if (age < 0xFFFF)
  {
    age++;
    streaming.resident.groups.d[residentID].age = uint16_t(age);
  }
  
  bool doUnload = age > streaming.ageThreshold;


  // if not unloaded due to traversal, test if unloaded due to no hits. This also
  // covers lod-clamp stream-out: groups no ray footprint demands stop being
  // opened by traversal (see `streamingDemandsFiner`) and age out above.
  uint hitValue = streaming.resident.groupHitFrames.d[residentID];
  // With the lod clamp traversal only opens what rays (or the camera floor) demand, so a group
  // it still opens must stay, unloading it would only have it requested again.
  if (useHitStreaming && !doUnload && !keep && (streaming.useHitStreamingLod == 0 || age > 2) &&
    int(streamingHitFrameDelta(hitValue)) > streaming.ageThreshold + 32)
  {
    // only unload fine-to-coarse: a group with resident finer (generating) groups
    // must stay, else the hit-frame inheritance chain of all coarser groups breaks.
    // the flag is computed in `stream_hit_propagation.comp.glsl` (one frame old).
    doUnload = (hitValue & STREAMING_HITFRAME_NO_FINER_BIT) != 0;
  }

  // detect if we are over the age limit and request the group to be unloaded
  if (doUnload)
  {    
    uint unloadOffset = atomicAdd(streamingRW.request.unloadCounter, 1);
    if (unloadOffset < streaming.request.maxUnloads) {
      streaming.request.unloadGeometryGroups.d[unloadOffset] = uvec2(geometryID, streaming.resident.groupIDs.d[residentID]);
    }
  }
}

#if USE_HIT_STREAMING

// reads the `index`-th 20-bit value from a tightly packed uint32 word stream
// (little-endian bit order). The word array is padded with one extra word so
// the `baseIndex + 1` read is always in bounds.
uint decode20Bit(uint32s_in bits, uint index)
{
  uint bitPos    = index * 20;
  uint baseIndex = bitPos >> 5;   // / 32
  uint bitOffset = bitPos & 31;   // % 32
  uint lo        = bits.d[baseIndex];
  uint hi        = bits.d[baseIndex + 1];
  // the value straddles into the next word only when it does not fit in `lo`
  hi = (bitOffset + 20 > 32) ? (hi << (32 - bitOffset)) : 0;
  return ((lo >> bitOffset) | hi) & 0xFFFFF;
}

void streamingRegisterHit(uint clusterID)
{
  // dereference the cluster from the resident cluster table
  uint groupID = streaming.resident.clusterGroups.d[clusterID];

  // all writers of a frame store the same value, no width tracking
  streaming.resident.groupHitFrames.d[groupID] = streamingMakeHitValue(0);
}

// Gates the stream-in request of a non-resident group from the propagated per-group
// demand (`groupDemand`), which carries the nearest recently-hit coarser ancestor's 
// recency down to this group, so the request can fire several levels below the 
// resident cut.
bool streamingTestOverlapHits(uint groupIndex, GroupOverlapInfos_in groupOverlapInfos, uint32s_inout groupDemand)
{
#if USE_HIT_STREAMING_LOD
  // lod clamping already gated this via `streamingDemandsFiner` (recency + width),
  // so reaching here means demanded; never suppress.
  return true;
#else
  // plain hit streaming: recency only. The demand window carries the grace, since
  // the BFS keeps a demanded group's stamp at most STREAMING_HITLOD_DEMAND_REFRESH old.
  // A group without coarser overlaps gets no demand from above, as in `streamingDemandsFiner`.
  return groupOverlapInfos.d[groupIndex].countCoarse == 0
         || streamingHitFrameDelta(groupDemand.d[groupIndex]) <= STREAMING_HITLOD_DEMAND_WINDOW;
#endif
}
#endif

#if USE_HIT_STREAMING_LOD

// registers a ray-cone hit (origin width + spread angle, world space), converted
// to object space so the value is shared across instances of the geometry.
//
// The width is evaluated at the group's bounding sphere, NOT the hit point — the
// same position the camera metric (`testForTraversal`) uses — so every ray from a
// given origin registers the identical width and marginal cut groups don't
// flicker. For converging cones (negative spread) the minimum over the sphere's
// distance interval is used, equally hit-point independent.
void streamingRegisterHitLod(uint clusterID, float coneWidthStart, float coneSpread, vec3 wRayOrigin, mat4x3 objectToWorld)
{
  uint groupID = streaming.resident.clusterGroups.d[clusterID];

  float uniformScale = max(max(length(objectToWorld[0]), length(objectToWorld[1])), length(objectToWorld[2]));

  TraversalMetric metric = streaming.resident.groups.d[groupID].group.d.traversalMetric;

  vec3  wSphereCenter  = objectToWorld * vec4(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ, 1.0);
  float sphereDistance = length(wSphereCenter - wRayOrigin);
  float sphereRadius   = metric.boundingSphereRadius * uniformScale;

  float wNear = coneWidthStart + coneSpread * max(sphereDistance - sphereRadius, 0.0);
  float wFar  = coneWidthStart + coneSpread * (sphereDistance + sphereRadius);

  // minimum width over the sphere interval, clamped at zero. Diverging cone: the
  // near endpoint. Converging cone: the far endpoint until the focus, then min<=0
  // clamps to zero = maximum demand, so the post-focus tail doesn't read coarser.
  float coneWidth = max(0.0, min(wNear, wFar));

  uint invWidth = streamingEncodeHitWidth(coneWidth / uniformScale);

  atomicMax(streaming.resident.groupHitFrames.d[groupID], streamingMakeHitValue(invWidth));
}

// Registers per-instance demand for detail finer than the camera metric would
// pick (a magnifying refraction/reflection). Unlike the per-group width above
// (shared across instances for residency), this scales only the demanding
// instance's traversal errorScale, so it refines and drops out of blas sharing
// while unmagnified siblings stay at camera detail.
//
// The scale is the ratio of the demanded cone width to the camera metric's
// acceptable width at the group (both at the bounding sphere, hit-point
// independent). scale < 1 means finer; it is floored to bound strong
// magnification and stored as a per-frame min (atomicMin on the float bits).
// Consumed in instance_classify_lod, traversal_run and traversal_run_groups.
void streamingRegisterInstanceDemand(uint instanceID,
                                     uint clusterID,
                                     float coneWidthStart,
                                     float coneSpread,
                                     vec3 wRayOrigin,
                                     vec3 wCameraPos,
                                     mat4x3 objectToWorld,
                                     float errorOverDistanceThreshold,
                                     float nearPlane,
                                     uint32s_inout instanceHitDemand)
{
  uint groupID = streaming.resident.clusterGroups.d[clusterID];

  float uniformScale = max(max(length(objectToWorld[0]), length(objectToWorld[1])), length(objectToWorld[2]));

  TraversalMetric metric = streaming.resident.groups.d[groupID].group.d.traversalMetric;

  vec3  wSphereCenter = objectToWorld * vec4(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ, 1.0);
  float sphereRadius  = metric.boundingSphereRadius * uniformScale;

  // demanded ray-cone width at the group (converging-cone minimum over the sphere
  // interval). The near endpoint uses the same nearPlane floor as the camera
  // metric below, so a primary ray yields exactly scale == 1 and never perturbs an
  // unmagnified instance.
  float rayDistance = length(wSphereCenter - wRayOrigin);
  float wNear       = coneWidthStart + coneSpread * max(rayDistance - sphereRadius, nearPlane);
  float wFar        = coneWidthStart + coneSpread * (rayDistance + sphereRadius);
  // clamp at the focus (min <= 0) = maximum demand, so an object past the focus
  // keeps demanding finest detail instead of the post-focus tail reading coarser.
  float coneWidth   = max(0.0, min(wNear, wFar));

  // camera metric's acceptable width at the group: errorOverDistanceThreshold is
  // the camera cone's spread, cameraDistance the near-plane-clamped distance to the
  // sphere. uniformScale cancels in the ratio, so world space is fine.
  float cameraDistance = max(nearPlane, length(wSphereCenter - wCameraPos) - sphereRadius);
  float cameraWidth    = errorOverDistanceThreshold * cameraDistance;

  // absolute errorScale the footprint demands (1 == camera cut, < 1 finer). Only
  // the finer direction is registered, so an unhit or coarser-hit instance keeps
  // the "no demand" sentinel and its errorScale is untouched by the min() combine.
  float target = coneWidth / max(cameraWidth, 1e-30);
  if (target < 1.0)
  {
    atomicMin(instanceHitDemand.d[instanceID], floatBitsToUint(max(target, STREAMING_HITLOD_INSTANCE_SCALE_MIN)));
  }
}

// true if a recent ray hit on the group's region demands detail finer than the
// group's error. Reads the flooded per-group demand (`groupDemand`), which summarizes
// this group's recent hits and, for a non-resident finer group, the demand inherited 
// from its coarser overlaps down to the resident hit cut — one read reaching many 
// levels below it.
//
// Clamps the lod traversal: a group is opened (rendered, or requested when not
// resident) only when rays need its finer content. Must match between
// traversal_run.comp.glsl and traversal_run_groups.comp.glsl.
bool streamingDemandsFiner(uint groupIndex, GroupOverlapInfos_in groupOverlapInfos, uint32s_inout groupDemand, float quadricError)
{
  if (groupOverlapInfos.d[groupIndex].countCoarse == 0)
  {
    // no coarser overlaps (coarsest level): never clamp, otherwise nothing
    // could render at all for a geometry without recent hits.
    return true;
  }

  uint demand = groupDemand.d[groupIndex];
  return streamingHitFrameDelta(demand) <= STREAMING_HITLOD_DEMAND_WINDOW
         && streamingDecodeHitWidth(demand) < quadricError * streaming.hitLodWidthFactor;
}

#endif

#endif  // STREAMING_GLSL
