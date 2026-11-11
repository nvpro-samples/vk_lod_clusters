/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  This compute shader handles updating the scene.
  Previous requests to load/unload have been completed and
  are provided for patching the scene.
  
  Effectively we are manipulating the geometries' 
  `streamingGroupAddresses` array that points to the resident
  memory location of a group (or tags it invalid).
  
  Furthermore when ray tracing is required we prepare building
  new CLAS for the loaded groups' clusters.
  
  After building is completed we run the `stream_move_new_clas.comp.glsl`
  to move them from temporary to final location.

  A thread represents a single patch operation, which takes care of
  one group.
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
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

////////////////////////////////////////////

layout(local_size_x=STREAM_UPDATE_SCENE_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_UPDATE_SCENE_WORKGROUP
  
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);  

  // works for both load and unload
  StreamingPatch spatch = streaming.update.patches.d[threadID];
  
  if (threadID < streaming.update.patchGroupsCount)
  {
    uint oldResidentID = 0;
    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
      Group group = Group_in(geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupID]).d;
      oldResidentID = group.residentID;
    }
    
    geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupID] = spatch.groupAddress;
    
    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
    #if STREAMING_DEBUG_ADDRESSES
      streaming.resident.groups.d[oldResidentID].group = Group_in(STREAMING_INVALID_ADDRESS_START);
    #endif
    }
    else
    {
      uint loadGroupIndex = threadID - streaming.update.patchUnloadGroupsCount;
      Group_in groupRef = Group_in(spatch.groupAddress);
      Group group = Group_in(groupRef).d;

      uint groupResidentID  = spatch.groupResidentID;
      groupRef.d.residentID = spatch.groupResidentID;
      groupRef.d.clusterResidentID = spatch.clusterResidentID;
    
      StreamingGroup residentGroup;
      residentGroup.geometryID   = spatch.geometryID;
      residentGroup.lodLevel     = spatch.lodLevel;
      residentGroup.age          = uint16_t(0);
      residentGroup.group        = groupRef;
    #if STREAMING_DEBUG_ADDRESSES
      if (uint64_t(streaming.resident.groups.d[groupResidentID].group) < STREAMING_INVALID_ADDRESS_START)
        streamingRW.request.errorUpdate = groupResidentID;
    #endif
      
      // update description in residency table
      streaming.resident.groups.d[groupResidentID] = residentGroup;

      // retain original groupID, used for unloading
      streaming.resident.groupIDs.d[groupResidentID] = spatch.groupID;

      // insert ourselves into the list of all active groups
      streaming.resident.activeGroups.d[streaming.update.loadActiveGroupsOffset + loadGroupIndex] = groupResidentID;
      
      // hit streaming (freshly streamed in data should be treated hit):
      // seed the frame stamp so the recency-gated stream-in keeps advancing
      // ~one level per streaming update. The width is seeded with *least*
      // demand (zero bits) — lod demand for the region comes from the coarser
      // overlap groups' real hit widths until this group is rendered and hit
      // itself; a maximum-demand seed would linger in the slowly relaxing
      // smoothed width and locally disable the clamp. This also overwrites
      // stale values of a previous occupant of the resident slot.
      streaming.resident.groupHitFrames.d[groupResidentID] =
          STREAMING_HITFRAME_NO_FINER_BIT
          | (STREAMING_HITFRAME_CYCLIC(streaming.frameIndex) << STREAMING_HITFRAME_FRAME_SHIFT);
      streaming.resident.groupHitWidths.d[groupResidentID] = 0;
      
      // We might have a bit of divergence here, but shouldn't be a mission critical issue
      
      // All new groups need to build new clusters.
      // These are built into scratch space first, and then moved to final locations.
      
      uint newBuildOffset = spatch.clasBuildOffset;
    #if TARGETS_RAY_TRACING
      // positions are streamed to CLAS-build scratch at a fixed per-group slot, tightly packed
      // per cluster (runningPosByteOffset)
      uint64_t groupPositionsBase =
          streaming.update.clasVerticesBuffer + uint64_t(loadGroupIndex) * streaming.update.clasVerticesGroupStride;
      uint runningPosByteOffset = 0;
    #endif
      for (uint c = 0; c < spatch.clusterCount; c++)
      {
        uint clusterResidentID = spatch.clusterResidentID + c;
        
        Cluster_in clusterRef = Cluster_in(spatch.groupAddress + Group_size + Cluster_size * c);
        streaming.resident.clusters.d[clusterResidentID] = uint64_t(clusterRef);
        
      #if TARGETS_RAY_TRACING
        // only for hit streaming
        streaming.resident.clusterGroups.d[clusterResidentID] = groupResidentID;
      
        Cluster cluster = clusterRef.d;
      
        ClasBuildInfo buildInfo;
        buildInfo.clusterID    = clusterResidentID;
        buildInfo.clusterFlags = 0;
        
        buildInfo.packed = 0;
        buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_triangleCount, cluster.triangleCountMinusOne+1);
        buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_vertexCount, cluster.vertexCountMinusOne+1);
        buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_indexType, 1);
        buildInfo.packed |= PACKED_FLAG(ClasBuildInfo_packed_positionTruncateBitCount, streaming.clasPositionTruncateBits);
        
        buildInfo.baseGeometryIndexAndFlags = 0;

        bool useGeometryIndices = (cluster.stateBits & CLUSTER_STATE_ALPHAMASKED_MIXED) != 0 || 
                                  (cluster.stateBits & CLUSTER_STATE_TWOSIDED_MIXED) != 0;
        
        buildInfo.indexBufferStride                 = uint16_t(1);
        buildInfo.vertexBufferStride                = uint16_t(4 * 3);
        buildInfo.geometryIndexAndFlagsBufferStride = uint16_t(0);
        buildInfo.opacityMicromapIndexBufferStride  = uint16_t(0);
        
        buildInfo.vertexBuffer = groupPositionsBase + uint64_t(runningPosByteOffset);
        buildInfo.indexBuffer  = uint64_t(Cluster_getTriangleIndices(Cluster_in(clusterRef)));
        
        buildInfo.geometryIndexAndFlagsBuffer = 0;
        buildInfo.opacityMicromapArray        = 0;
        buildInfo.opacityMicromapIndexBuffer  = 0;

      #if HAS_ALPHA_TEST
        if (useGeometryIndices)
        {
          uint base = atomicAdd(streamingRW.update.newClasGeometryIndicesTaskCounter, 1);

          uint64_t geometryIndicesBuffer = uint64_t(streaming.update.newClasGeometryIndices) + uint64_t(base * CLUSTER_TRIANGLE_COUNT * 4);

          buildInfo.baseGeometryIndexAndFlags           = 0u;
          buildInfo.geometryIndexAndFlagsBufferStride   = uint16_t(4);
          buildInfo.geometryIndexAndFlagsBuffer = geometryIndicesBuffer;

          // we hijack the first values of geometry indices array to store a reference to the cluster,
          // so that the `stream_update_clas_geometry_indices.comp.glsl` kernel can fill the per-triangle
          // information accordingly.
          uint64s_inout clusterInfos = uint64s_inout(geometryIndicesBuffer);
          clusterInfos.d[0] = uint64_t(clusterRef);
        }
        else
      #endif
        {
          buildInfo.baseGeometryIndexAndFlags = HAS_ALPHA_TEST != 0 && (cluster.stateBits & CLUSTER_STATE_ALPHAMASKED) != 0 ? 1 : ClasGeometryFlag_OPAQUE_BIT_NV;
          buildInfo.baseGeometryIndexAndFlags |= (cluster.stateBits & CLUSTER_STATE_TWOSIDED) != 0 ? ClasGeometryFlag_CULL_DISABLE_BIT_NV : 0;
        }

        streaming.update.newClasBuilds.d[newBuildOffset + c]      = buildInfo;
        streaming.update.newClasResidentIDs.d[newBuildOffset + c] = clusterResidentID;

        runningPosByteOffset += (cluster.vertexCountMinusOne + 1) * 12;
      #endif
      }
    }
  }
#if 1  // relevant to USE_BLAS_CACHING
  if (threadID < streaming.update.patchCachedBlasCount)
  {
    StreamingGeometryPatch sgpatch = streaming.update.geometryPatches.d[threadID];
    uint discreteLodLevel          = sgpatch.discreteLodLevel;
    uint geometryID                = sgpatch.geometryID;
    geometries[geometryID].discreteLodLevel  = uint8_t(discreteLodLevel);
    streaming.cachedBlasAddresses.d[geometryID * GEOMETRY_CACHED_BLAS_STRIDE
                                    + GEOMETRY_CACHED_BLAS_SLOT(geometries[geometryID].lodLevelsCount,
                                                                sgpatch.discreteLodLevel)] = sgpatch.cachedBlasAddress;
  }
#endif
}

