/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>
#include <math.h>

#include "scene_kdop.hpp"

namespace lodclusters {

const glm::vec3* kdopGetAxes()
{
  static const glm::vec3 s_axes[KDOP_AXIS_COUNT] = {
      // face
      {1, 0, 0},
      {0, 1, 0},
      {0, 0, 1},
      // corner
      {1, 1, 1},
      {1, 1, -1},
      {1, -1, 1},
      {1, -1, -1},
      // edge
      {1, 1, 0},
      {1, -1, 0},
      {1, 0, 1},
      {1, 0, -1},
      {0, 1, 1},
      {0, 1, -1},
  };

  return s_axes;
}

void KDop::reset(const glm::mat3& frameInit)
{
  for(uint32_t i = 0; i < KDOP_AXIS_COUNT; i++)
  {
    lo[i] = FLT_MAX;
    hi[i] = -FLT_MAX;
  }

  frame = frameInit;
}

void KDopAccumulator::init(const glm::mat3& frame)
{
  // `dot(axis, frame * position) == dot(transpose(frame) * axis, position)`,
  // moves the rotation off the hot loop
  glm::mat3 frameT = glm::transpose(frame);

  for(uint32_t i = 0; i < KDOP_AXIS_COUNT; i++)
  {
    objectAxes[i] = frameT * kdopGetAxes()[i];

    lo[i] = FLT_MAX;
    hi[i] = -FLT_MAX;
  }
}

void KDopAccumulator::mergeInto(KDop& dst) const
{
  for(uint32_t i = 0; i < KDOP_AXIS_COUNT; i++)
  {
    dst.lo[i] = std::min(dst.lo[i], lo[i]);
    dst.hi[i] = std::max(dst.hi[i], hi[i]);
  }
}

//////////////////////////////////////////////////////////////////////////

// Cyclic Jacobi for a symmetric 3x3, eigenvectors returned as columns of `vectors`.
// Builds the rotation explicitly and multiplies it out, a handful of 3x3 products
// per geometry, rather than hand unrolled update rules.
static void jacobiEigen(const glm::mat3& input, glm::mat3& vectors, glm::vec3& values)
{
  glm::mat3 a = input;
  glm::mat3 v = glm::mat3(1.0f);

  for(uint32_t sweep = 0; sweep < 16; sweep++)
  {
    // glm is column major, `a[column][row]`, `a` stays symmetric
    float offDiagonal = fabsf(a[1][0]) + fabsf(a[2][0]) + fabsf(a[2][1]);

    if(offDiagonal <= 1e-20f)
      break;

    for(uint32_t p = 0; p < 2; p++)
    {
      for(uint32_t q = p + 1; q < 3; q++)
      {
        float apq = a[q][p];

        if(fabsf(apq) <= 1e-20f)
          continue;

        float theta = (a[q][q] - a[p][p]) / (2.0f * apq);
        float t     = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
        float c     = 1.0f / sqrtf(t * t + 1.0f);
        float s     = t * c;

        glm::mat3 j = glm::mat3(1.0f);
        j[p][p]     = c;
        j[q][q]     = c;
        j[q][p]     = s;
        j[p][q]     = -s;

        a = glm::transpose(j) * a * j;
        v = v * j;
      }
    }
  }

  values  = glm::vec3(a[0][0], a[1][1], a[2][2]);
  vectors = v;
}

glm::mat3 kdopComputeFrame(const glm::vec3* positions, size_t positionCount, const glm::uvec3* triangles, size_t triangleCount)
{
  // single pass in double precision, accumulated about the origin and shifted
  // to the area weighted centroid at the end
  double     totalArea = 0.0;
  glm::dvec3 areaCenter(0.0);
  glm::dmat3 moments(0.0);

  for(size_t t = 0; t < triangleCount; t++)
  {
    glm::uvec3 indices = triangles[t];

    if(indices.x >= positionCount || indices.y >= positionCount || indices.z >= positionCount)
      continue;

    glm::dvec3 a = glm::dvec3(positions[indices.x]);
    glm::dvec3 b = glm::dvec3(positions[indices.y]);
    glm::dvec3 c = glm::dvec3(positions[indices.z]);

    glm::dvec3 edge0 = b - a;
    glm::dvec3 edge1 = c - a;

    double area = 0.5 * glm::length(glm::cross(edge0, edge1));

    if(!(area > 0.0))
      continue;

    glm::dvec3 center = (a + b + c) / 3.0;

    // covariance of a uniform distribution over the triangle, about its own centroid.
    // with `x = a + u * edge0 + v * edge1` over the unit simplex, E[u*u] = E[v*v] = 1/6
    // and E[u*v] = 1/12, which about the centroid gives 1/18 and -1/36.
    glm::dmat3 local = (glm::outerProduct(edge0, edge0) + glm::outerProduct(edge1, edge1)) / 18.0
                       - (glm::outerProduct(edge0, edge1) + glm::outerProduct(edge1, edge0)) / 36.0;

    totalArea += area;
    areaCenter += center * area;
    moments += (local + glm::outerProduct(center, center)) * area;
  }

  if(!(totalArea > 0.0))
    return glm::mat3(1.0f);

  areaCenter /= totalArea;

  // parallel axis shift onto the area weighted centroid
  glm::mat3 covariance = glm::mat3(moments - glm::outerProduct(areaCenter, areaCenter) * totalArea);

  glm::mat3 vectors;
  glm::vec3 values;
  jacobiEigen(covariance, vectors, values);

  // order by descending eigenvalue
  uint32_t order[3] = {0, 1, 2};

  for(uint32_t i = 0; i < 2; i++)
  {
    for(uint32_t j = i + 1; j < 3; j++)
    {
      if(values[order[j]] > values[order[i]])
        std::swap(order[i], order[j]);
    }
  }

  glm::vec3 axisX = vectors[order[0]];
  glm::vec3 axisY = vectors[order[1]];
  glm::vec3 axisZ = vectors[order[2]];

  // a flat or symmetric mesh can leave an axis degenerate, rebuild rather than trust it
  float lengthX = glm::length(axisX);
  float lengthY = glm::length(axisY);

  if(!(lengthX > 1e-12f) || !(lengthY > 1e-12f))
    return glm::mat3(1.0f);

  axisX = axisX / lengthX;
  axisY = glm::normalize(axisY - axisX * glm::dot(axisX, axisY));
  axisZ = glm::cross(axisX, axisY);

  if(!(glm::dot(axisX, axisX) > 0.0f) || !(glm::dot(axisY, axisY) > 0.0f) || !(glm::dot(axisZ, axisZ) > 0.0f))
    return glm::mat3(1.0f);

  // rows are the principal axes, so `frame * position` projects onto them
  return glm::transpose(glm::mat3(axisX, axisY, axisZ));
}

//////////////////////////////////////////////////////////////////////////

void kdopBuildHull(KDopHull& hull, const KDop& dop, const glm::vec3& boxLo, const glm::vec3& boxHi)
{
  hull.vertexCount = 0;

  if(!dop.isValid())
    return;

  // everything is brought to object space, the slab planes via the inverse frame, so the
  // resulting vertices need no transform afterwards. normals are unit length, which keeps
  // the determinant and containment thresholds meaningful.
  const uint32_t planeCount = KDOP_AXIS_COUNT * 2 + 6;

  glm::vec3 extent  = boxHi - boxLo;
  float     maxSide = std::max(extent.x, std::max(extent.y, extent.z));

  // slack on the containment test, without it rounding rejects vertices lying exactly
  // on a plane and the polytope opens up
  float containTolerance = maxSide * 1e-5f + FLT_MIN;
  float mergeDistance    = maxSide * 1e-4f;
  float mergeDistanceSq  = mergeDistance * mergeDistance;

  // The intersection rounds, and the merge below drops a vertex that sits within
  // `mergeDistance` of one already kept, either of which can pull the hull inside the
  // geometry and underbound thin geometry once it is rotated. Pushing every plane out by
  // that much contains the original polytope grown by the same radius, which absorbs both.
  float dilate = mergeDistance + containTolerance;

  // `dot(normal, x) <= distance`
  glm::vec3 planeNormals[planeCount];
  float     planeDistances[planeCount];

  glm::mat3 frameT = glm::transpose(dop.frame);

  for(uint32_t i = 0; i < KDOP_AXIS_COUNT; i++)
  {
    glm::vec3 axis      = frameT * kdopGetAxes()[i];
    float     axisScale = 1.0f / glm::length(axis);

    planeNormals[i * 2 + 0]   = axis * axisScale;
    planeDistances[i * 2 + 0] = dop.hi[i] * axisScale + dilate;
    planeNormals[i * 2 + 1]   = -axis * axisScale;
    planeDistances[i * 2 + 1] = -dop.lo[i] * axisScale + dilate;
  }

  for(uint32_t a = 0; a < 3; a++)
  {
    glm::vec3 axis = glm::vec3(0);
    axis[a]        = 1.0f;

    planeNormals[KDOP_AXIS_COUNT * 2 + a * 2 + 0]   = axis;
    planeDistances[KDOP_AXIS_COUNT * 2 + a * 2 + 0] = boxHi[a] + dilate;
    planeNormals[KDOP_AXIS_COUNT * 2 + a * 2 + 1]   = -axis;
    planeDistances[KDOP_AXIS_COUNT * 2 + a * 2 + 1] = -boxLo[a] + dilate;
  }

  bool overflow = false;

  for(uint32_t i = 0; i < planeCount && !overflow; i++)
  {
    for(uint32_t j = i + 1; j < planeCount && !overflow; j++)
    {
      for(uint32_t k = j + 1; k < planeCount && !overflow; k++)
      {
        glm::vec3 crossJK = glm::cross(planeNormals[j], planeNormals[k]);
        float     det     = glm::dot(planeNormals[i], crossJK);

        // triple product of unit normals, near zero means the planes share a direction
        if(fabsf(det) < 1e-6f)
          continue;

        glm::vec3 point = (crossJK * planeDistances[i] + glm::cross(planeNormals[k], planeNormals[i]) * planeDistances[j]
                           + glm::cross(planeNormals[i], planeNormals[j]) * planeDistances[k])
                          / det;

        bool inside = true;

        for(uint32_t m = 0; m < planeCount; m++)
        {
          if(glm::dot(planeNormals[m], point) > planeDistances[m] + containTolerance)
          {
            inside = false;
            break;
          }
        }

        if(!inside)
          continue;

        bool duplicate = false;

        for(uint32_t v = 0; v < hull.vertexCount; v++)
        {
          glm::vec3 delta = hull.vertices[v] - point;

          if(glm::dot(delta, delta) <= mergeDistanceSq)
          {
            duplicate = true;
            break;
          }
        }

        if(duplicate)
          continue;

        if(hull.vertexCount >= KDOP_MAX_HULL_VERTICES)
        {
          overflow = true;
          break;
        }

        hull.vertices[hull.vertexCount++] = point;
      }
    }
  }

  if(overflow)
  {
    // dropping vertices would make the set non conservative, falling back to the bbox
    // corners stays valid and just reproduces what the bbox alone would give
    assert(0 && "kdop hull vertex budget exceeded");

    hull.vertexCount = 0;

    for(uint32_t corner = 0; corner < 8; corner++)
    {
      glm::bvec3 weight((corner & 1) != 0, (corner & 2) != 0, (corner & 4) != 0);

      hull.vertices[hull.vertexCount++] = glm::mix(boxLo, boxHi, weight);
    }
  }
}

}  // namespace lodclusters
