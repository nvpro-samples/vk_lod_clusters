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

#include "shaderio_core.h"

#ifndef _SHADERIO_SCENE_H_
#define _SHADERIO_SCENE_H_

#ifdef __cplusplus
namespace shaderio {
using namespace glm;

enum ClusterAttributeBits
{
  CLUSTER_ATTRIBUTE_VERTEX_NORMAL  = 1,
  CLUSTER_ATTRIBUTE_VERTEX_UV      = 2,
  CLUSTER_ATTRIBUTE_VERTEX_TANGENT = 4,
};

#else

#define CLUSTER_ATTRIBUTE_VERTEX_NORMAL 1
#define CLUSTER_ATTRIBUTE_VERTEX_UV 2
#define CLUSTER_ATTRIBUTE_VERTEX_TANGENT 4

#ifndef CLUSTER_VERTEX_COUNT
#define CLUSTER_VERTEX_COUNT 32
#endif

#ifndef CLUSTER_TRIANGLE_COUNT
#define CLUSTER_TRIANGLE_COUNT 32
#endif

#endif

#define SHADERIO_ORIGINAL_MESH_GROUP 0xffffffffu
#define SHADERIO_MAX_LOD_LEVELS 32
#define SHADERIO_MAX_GROUP_CLUSTERS 128

struct BBox
{
  vec3 lo;
  vec3 hi;
  // relevant to cluster's triangles
  float shortestEdge;
  float longestEdge;
};
BUFFER_REF_DECLARE(BBox_in, BBox, readonly, 16);
BUFFER_REF_DECLARE_ARRAY(BBoxes_in, BBox, readonly, 16);

// A cluster contains a small number of triangles and vertices.
// It is always part of a group.
struct Cluster
{
  uint8_t triangleCountMinusOne;
  uint8_t vertexCountMinusOne;
  uint8_t lodLevel;
  uint8_t groupChildIndex;

  uint8_t  attributeBits;
  uint8_t  localMaterialID;  // if 0xFF then a per-triangle index is used
  uint16_t reserved;

  uint32_t vertices;  // vec3 positions[vertexCount], followed by optional per-vertex attributes
  uint32_t indices;   // uint8_t indices[triangleCount * 3], followed by optional per-triangle material index
};
BUFFER_REF_DECLARE(Cluster_in, Cluster, , 16);
BUFFER_REF_DECLARE_ARRAY(Clusters_inout, Cluster, , 16);
BUFFER_REF_DECLARE_SIZE(Cluster_size, Cluster, 16);

#ifndef __cplusplus
vec3s_in Cluster_getVertexPositions(Cluster_in cluster)
{
  return vec3s_in(uint64_t(cluster) + cluster.d.vertices);
}
uint32s_in Cluster_getVertexNormals(Cluster_in cluster)
{
  return uint32s_in(uint64_t(cluster) + (cluster.d.vertices + 4 * 3 * (cluster.d.vertexCountMinusOne + 1)));
}
vec2s_in Cluster_getVertexUVs(Cluster_in cluster)
{
  uint32_t elems = (cluster.d.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_NORMAL) == 0 ? 3 : 4;
  return vec2s_in(uint64_t(cluster) + (((cluster.d.vertices + 4 * elems * (cluster.d.vertexCountMinusOne + 1)) + 7) & ~7));
}
uint16s_in Cluster_getVertexTangents(Cluster_in cluster)
{
  // tangents only exist if normals and uvs do as well
  return uint16s_in(uint64_t(Cluster_getVertexUVs(cluster)) + ((cluster.d.vertexCountMinusOne + 1) * 4 * 2));
}
uint8s_in Cluster_getTriangleIndices(Cluster_in cluster)
{
  return uint8s_in(uint64_t(cluster) + cluster.d.indices);
}
uint8s_in Cluster_getTriangleMaterials(Cluster_in cluster)
{
  return uint8s_in(uint64_t(cluster) + (cluster.d.indices + 3 * (cluster.d.triangleCountMinusOne + 1)));
}
#endif

// A group contains multiple clusters that are the result of
// a common mesh decimation operation. Clusters within a group
// are watertight to each other. Groups are always streamed in
// completely, which simplifies the streaming management.

struct TraversalMetric
{
  // must not change order, due to nvlclusterlod::Node
  // scalar by design, avoid hiccups with packing
  float boundingSphereX;
  float boundingSphereY;
  float boundingSphereZ;
  float boundingSphereRadius;
  float maxQuadricError;
};

struct Group
{
  // streaming: global unique id given on load
  //            clusters array starts directly after group
  // preloaded: local id within geometry
  uint32_t residentID;
  uint32_t clusterResidentID;

  uint16_t lodLevel;
  uint16_t clusterCount;  // 128 max

  TraversalMetric traversalMetric;
};

BUFFER_REF_DECLARE(Group_in, Group, , 16);
BUFFER_REF_DECLARE_ARRAY(Groups_in, Group, , 16);
BUFFER_REF_DECLARE_SIZE(Group_size, Group, 32);

#ifndef __cplusplus
uint Group_getGeneratingGroup(Group_in group, uint clusterIndex)
{
  return uint32s_in(uint64_t(group) + uint32_t(Group_size + Cluster_size * group.d.clusterCount)).d[clusterIndex];
}
BBox Group_getClusterBBox(Group_in group, uint clusterIndex)
{
  return BBoxes_in(uint64_t(group)
                   + uint32_t(Group_size + Cluster_size * group.d.clusterCount + (((4 * group.d.clusterCount) + 15) & ~15)))
      .d[clusterIndex];
}
Group_in Cluster_getGroup(Cluster_in cluster)
{
  return Group_in(uint64_t(cluster) - uint32_t(cluster.d.groupChildIndex * Cluster_size + Group_size));
}
BBox Cluster_getBBox(Cluster_in cluster)
{
  return Group_getClusterBBox(Cluster_getGroup(cluster), cluster.d.groupChildIndex);
}
#endif

#ifdef __cplusplus
// must match `nvclusterlod::InteriorNode`
struct NodeRange
{
  uint32_t isGroup : 1;
  uint32_t childOffset : 26;
  uint32_t childCountMinusOne : 5;
};

// must match `nvclusterlod::LeafNode`
struct GroupRange
{
  uint32_t isGroup : 1;
  uint32_t groupIndex : 23;
  uint32_t groupClusterCountMinusOne : 8;
};
#endif

// must match `nvclusterlod::Node`
struct Node
{
#ifdef __cplusplus
  union
  {
    NodeRange  nodeRange;
    GroupRange groupRange;
  };
#else
  uint32_t packed;

#define Node_packed_isGroup 0 : 1

#define Node_packed_nodeChildOffset 1 : 26
#define Node_packed_nodeChildCountMinusOne 27 : 5

#define Node_packed_groupIndex 1 : 23
#define Node_packed_groupClusterCountMinusOne 24 : 8

#endif
  // use scalar to avoid glsl alignment hiccups
  TraversalMetric traversalMetric;
};
BUFFER_REF_DECLARE_ARRAY(Nodes_in, Node, readonly, 8);

struct LodLevel
{
  float    minBoundingSphereRadius;
  float    minMaxQuadricError;
  uint32_t groupOffset;
  uint32_t groupCount;
  uint32_t clusterOffset;
  uint32_t clusterCount;
};
BUFFER_REF_DECLARE_ARRAY(LodLevels_inout, LodLevel, , 8);

struct Geometry
{
  uint32_t instancesOffset;
  uint32_t instancesCount;
  uint8_t  lodLevelsCount;
  uint8_t  cachedBlasLodLevel;  // for USE_BLAS_CACHING

  // lowest detail data is always available
  uint16_t lowDetailTriangles;
  uint32_t lowDetailClusterID;
  uint64_t lowDetailBlasAddress;

  // only for USE_BLAS_CACHING in streaming
  uint64_t cachedBlasAddress;

  // object space geometry bbox
  BBox bbox;

  BUFFER_REF(LodLevels_inout) lodLevels;

  // lod hierarchy traversal
  BUFFER_REF(Nodes_in) nodes;
  BUFFER_REF(BBoxes_in) nodeBboxes;

  // streaming (null if preloaded)
  // provides memory address of a resident group.
  //
  // Note this 64-bit value uses a special encoding.
  // only addresses < STREAMING_INVALID_ADDRESS_BEGIN can be dereferenced.
  BUFFER_REF(uint64s_inout) streamingGroupAddresses;

  // preloaded (null if streaming)
  // clusters
  BUFFER_REF(uint64s_in) preloadedGroups;
  BUFFER_REF(uint64s_in) preloadedClusters;
  // for ray tracing
  BUFFER_REF(uint64s_in) preloadedClusterClasAddresses;
  BUFFER_REF(uint32s_in) preloadedClusterClasSizes;
};
BUFFER_REF_DECLARE(Geometry_in, Geometry, readonly, 16);
BUFFER_REF_DECLARE(Geometry_inout, Geometry, , 16);

struct RenderInstance
{
  mat4 worldMatrix;
  mat4 worldMatrixI;

  uint32_t geometryID;
  uint32_t materialID;
  float    maxLodLevelRcp;
  uint32_t packedColor;
};
BUFFER_REF_DECLARE_ARRAY(RenderInstances_in, RenderInstance, readonly, 16);

#ifdef __cplusplus
}
#endif
#endif
