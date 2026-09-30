/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <float.h>
#include <algorithm>

#include <glm/glm.hpp>

namespace lodclusters {

// A 26-DOP (13 slabs) per geometry, stored in an oriented frame.
// Used to build tighter per-instance AABBs than `instanceMatrix * geometryBbox` allows.
// Host only, not uploaded.
//
// References for the standard pieces used here:
// - k-DOPs: Klosowski et al., "Efficient Collision Detection Using Bounding Volume
//   Hierarchies of k-DOPs", IEEE TVCG 1998
// - area weighted triangle covariance for the oriented frame: Gottschalk et al.,
//   "OBBTree: A Hierarchical Structure for Rapid Interference Detection", SIGGRAPH 1996
//   https://gamma.cs.unc.edu/OBB/
// - symmetric eigen solve: the classic cyclic Jacobi rotation from Numerical Recipes
// The per-triangle covariance term in `kdopComputeFrame` is derived inline, see there.

static const uint32_t KDOP_AXIS_COUNT = 13;

// `V <= 2 * F - 4` bounds the 26 slab planes plus the 6 bbox planes at 60 vertices,
// rest is slack for near degenerate slabs
static const uint32_t KDOP_MAX_HULL_VERTICES = 64;

// 3 face, 4 corner, 6 edge directions. un-normalized, the scale cancels when the
// half-spaces are intersected. changing the table needs a `geoVersion` bump.
const glm::vec3* kdopGetAxes();

// slab distances along `kdopGetAxes()`, measured in `frame` space
struct KDop
{
  float lo[KDOP_AXIS_COUNT];
  float hi[KDOP_AXIS_COUNT];

  // object to dop space, orthonormal. its rows are the principal axes, which lets the
  // fixed cube-symmetric directions adapt to elongated or diagonally oriented geometry.
  glm::mat3 frame;

  void reset(const glm::mat3& frameInit);

  // false until a position was added
  bool isValid() const { return lo[0] <= hi[0]; }
};

// keeps the axes pre-transformed to object space, so `add` is just 13 dot products
struct KDopAccumulator
{
  glm::vec3 objectAxes[KDOP_AXIS_COUNT];
  float     lo[KDOP_AXIS_COUNT];
  float     hi[KDOP_AXIS_COUNT];

  void init(const glm::mat3& frame);

  inline void add(const glm::vec3& objectPosition)
  {
    for(uint32_t i = 0; i < KDOP_AXIS_COUNT; i++)
    {
      float distance = glm::dot(objectAxes[i], objectPosition);

      lo[i] = std::min(lo[i], distance);
      hi[i] = std::max(hi[i], distance);
    }
  }

  // extends `dst`, must use the same frame
  void mergeInto(KDop& dst) const;
};

// Eigenvectors of the area weighted triangle covariance, descending by eigenvalue.
// Area weighting avoids biasing the frame towards densely tessellated regions.
// Identity if the input has no area.
glm::mat3 kdopComputeFrame(const glm::vec3* positions, size_t positionCount, const glm::uvec3* triangles, size_t triangleCount);

// the point set per-instance AABBs are built from
struct KDopHull
{
  uint32_t  vertexCount = 0;
  glm::vec3 vertices[KDOP_MAX_HULL_VERTICES];
};

// Intersects the 26 slab half-spaces with the 6 of the object bbox and returns the
// resulting polytope's vertices, in object space.
// Bounds must be taken over these, the geometry's support points along the slab
// directions are not conservative under rotation.
// The bbox planes matter because the slabs are measured in the oriented frame, so without
// them the polytope can stick out past the bbox and end up worse than it for an
// axis aligned instance.
// Brute force over all C(32,3) plane triples, cheap next to the rest of the load.
// The result is dilated by a small fraction of the extent, so rounding in the intersection
// cannot leave it underbounding the geometry.
void kdopBuildHull(KDopHull& hull, const KDop& dop, const glm::vec3& boxLo, const glm::vec3& boxHi);

}  // namespace lodclusters
