/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>

#include <fmt/format.h>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>

#include "scene.hpp"

namespace lodclusters {

// interleave the low 16 bits of x and y
static uint32_t part1By1(uint32_t v)
{
  v &= 0x0000ffffu;
  v = (v ^ (v << 8)) & 0x00ff00ffu;
  v = (v ^ (v << 4)) & 0x0f0f0f0fu;
  v = (v ^ (v << 2)) & 0x33333333u;
  v = (v ^ (v << 1)) & 0x55555555u;
  return v;
}

uint32_t Scene::getPartitionGridBaseCells(const ScenePartitionConfig& partitionConfig) const
{
  uint32_t cells = std::max(1u, uint32_t(std::ceil(std::sqrt(double(m_instances.size())
                                                             / double(std::max(1u, partitionConfig.derivedTargetInstances))))));
  // every base cell can become a partition
  uint32_t budgetCells = std::max(1u, uint32_t(std::sqrt(double(getPartitionBudget(partitionConfig)))));
  return std::min(cells, budgetCells);
}

glm::vec2 Scene::getPartitionGridPos(const shaderio::BBox& instanceBbox, uint32_t perAxis) const
{
  // grid over xz, gltf scenes are y up
  const glm::vec2 gridLo(m_gridBbox.lo.x, m_gridBbox.lo.z);
  const glm::vec2 gridExtent = glm::max(glm::vec2(m_gridBbox.hi.x, m_gridBbox.hi.z) - gridLo, glm::vec2(1e-6f));

  glm::vec2 center((instanceBbox.lo.x + instanceBbox.hi.x) * 0.5f, (instanceBbox.lo.z + instanceBbox.hi.z) * 0.5f);
  return glm::clamp((center - gridLo) / gridExtent, glm::vec2(0.0f), glm::vec2(0.9999f)) * glm::vec2(float(perAxis));
}

uint32_t Scene::buildPartitionsGrid(const ScenePartitionConfig& partitionConfig)
{
  uint32_t partitionCount = 0;

  const bool isDerived = partitionConfig.layout == ScenePartitionConfig::LAYOUT_DERIVED_GRID;

  uint32_t perAxis = isDerived ? getPartitionGridBaseCells(partitionConfig) : std::max(1u, partitionConfig.gridCellsPerAxis);

  std::vector<uint32_t> cellIndices(m_instances.size());
  std::vector<uint32_t> cellCounts;

  while(true)
  {
    cellCounts.assign(size_t(perAxis) * size_t(perAxis), 0);

    nvutils::parallel_batches<1024>(m_instances.size(), [&](uint64_t i) {
      glm::uvec2 cell = glm::uvec2(getPartitionGridPos(m_instances[i].bbox, perAxis));
      cellIndices[i]  = cell.x + cell.y * perAxis;
    });

    uint32_t maxInstances = 0;
    for(uint32_t cellIndex : cellIndices)
    {
      cellCounts[cellIndex]++;
      maxInstances = std::max(maxInstances, cellCounts[cellIndex]);
    }

    // a cell splits into four, so the used count can at most quadruple
    uint32_t usedCells = 0;
    for(uint32_t count : cellCounts)
    {
      usedCells += count ? 1 : 0;
    }

    // Refine until the worst partition fits, but the occupancy chase runs away on
    // clustered scenes, where it costs build time for partitions that are mostly empty.
    // The partition count is what the build cost tracks, so it is the binding budget.
    bool fitsMaxPartitions = partitionConfig.maxPartitions == 0
                             || uint64_t(perAxis) * uint64_t(perAxis) * 4 <= uint64_t(partitionConfig.maxPartitions);
    bool fitsPartitionCount =
        partitionConfig.derivedMaxCount == 0 || uint64_t(usedCells) * 4 <= uint64_t(partitionConfig.derivedMaxCount);
    // instances at one position never split, the fixed grid's ui limit stops those
    bool fitsGrid = perAxis * 2 <= 4096;
    if(!isDerived || maxInstances <= partitionConfig.derivedMaxInstances || !fitsMaxPartitions || !fitsPartitionCount || !fitsGrid)
    {
      break;
    }
    perAxis *= 2;
  }

  m_partitionsPerAxis = perAxis;

  // Empty cells are dropped, the remaining ones become the partitions. They are numbered
  // along a Z-order curve rather than in grid raster order, so that partitions close in
  // space are close in index. Everything downstream is indexed by partition: the instance
  // ordering, and with it the blocks the early-out compacts. In raster order two
  // partitions adjacent in a block can be a full grid row apart, and a camera moving along
  // z would dirty cells striding by `perAxis`, touching a different block each time.
  std::vector<uint32_t> cellPartitions(cellCounts.size(), ~0u);
  {
    std::vector<std::pair<uint32_t, uint32_t>> occupied;  // morton, cellIndex
    occupied.reserve(m_instances.size() / 16 + 1);
    for(size_t cellIndex = 0; cellIndex < cellCounts.size(); cellIndex++)
    {
      if(cellCounts[cellIndex])
      {
        uint32_t x = uint32_t(cellIndex) % perAxis;
        uint32_t y = uint32_t(cellIndex) / perAxis;
        occupied.push_back({part1By1(x) | (part1By1(y) << 1), uint32_t(cellIndex)});
      }
    }
    std::sort(occupied.begin(), occupied.end());

    uint32_t partitionIndex = 0;
    for(const auto& it : occupied)
    {
      cellPartitions[it.second] = partitionIndex++;
    }
    partitionCount = partitionIndex;
  }

  nvutils::parallel_batches<1024>(m_instances.size(),
                                  [&](uint64_t i) { m_instancePartitions[i] = cellPartitions[cellIndices[i]]; });

  return partitionCount;
}

// Fixed size cells, aligned to the shifted world origin rather than the scene bounds, so the
// cells stay put when the scene or its grid copies change. Unbounded, so the occupied cells
// are numbered along a z curve over their full 32 bit range instead of a dense grid.
uint32_t Scene::buildPartitionsCells(const ScenePartitionConfig& partitionConfig)
{
  const glm::vec2 origin(partitionConfig.gridOrigin.x, partitionConfig.gridOrigin.z);
  const float     cellSize = partitionConfig.gridCellSize;

  std::vector<glm::ivec2> cells(m_instances.size());
  nvutils::parallel_batches<1024>(m_instances.size(), [&](uint64_t i) {
    const Instance& instance = m_instances[i];

    glm::vec2 center((instance.bbox.lo.x + instance.bbox.hi.x) * 0.5f, (instance.bbox.lo.z + instance.bbox.hi.z) * 0.5f);
    cells[i] =
        glm::ivec2(glm::clamp(glm::floor((center - origin) / cellSize), glm::vec2(-float(1 << 30)), glm::vec2(float(1 << 30))));
  });

  glm::ivec2 cellMin(INT32_MAX);
  for(size_t i = 0; i < cells.size(); i++)
  {
    cellMin = glm::min(cellMin, cells[i]);
  }

  // interleave the 32 bits of x and y
  auto part1By1 = [](uint64_t v) {
    v &= 0x00000000ffffffffull;
    v = (v ^ (v << 16)) & 0x0000ffff0000ffffull;
    v = (v ^ (v << 8)) & 0x00ff00ff00ff00ffull;
    v = (v ^ (v << 4)) & 0x0f0f0f0f0f0f0f0full;
    v = (v ^ (v << 2)) & 0x3333333333333333ull;
    v = (v ^ (v << 1)) & 0x5555555555555555ull;
    return v;
  };

  std::vector<uint64_t> keys(m_instances.size());
  nvutils::parallel_batches<1024>(m_instances.size(), [&](uint64_t i) {
    glm::uvec2 cell = glm::uvec2(cells[i] - cellMin);
    keys[i]         = part1By1(cell.x) | (part1By1(cell.y) << 1);
  });

  std::vector<uint64_t> occupied = keys;
  std::sort(occupied.begin(), occupied.end());
  occupied.erase(std::unique(occupied.begin(), occupied.end()), occupied.end());

  nvutils::parallel_batches<1024>(m_instances.size(), [&](uint64_t i) {
    m_instancePartitions[i] = uint32_t(std::lower_bound(occupied.begin(), occupied.end(), keys[i]) - occupied.begin());
  });

  return uint32_t(occupied.size());
}

// Every grid copy gets its own set of node groups, the instances below no matching node
// share a single partition after them.
uint32_t Scene::buildPartitionsNodeGroups()
{
  const size_t   originalCount = m_originalInstanceCount;
  const uint32_t copiesCount   = uint32_t(m_instances.size() / originalCount);
  const uint32_t groupsCount   = m_nodeGroupCount * copiesCount;

  std::vector<uint32_t> keys(m_instances.size());
  std::vector<uint32_t> remap(size_t(groupsCount) + 1, 0);
  for(size_t i = 0; i < m_instances.size(); i++)
  {
    uint32_t group = m_instanceNodeGroups.empty() ? ~0u : m_instanceNodeGroups[i % originalCount];
    keys[i]        = group != ~0u ? uint32_t(i / originalCount) * m_nodeGroupCount + group : groupsCount;
    remap[keys[i]] = 1;
  }

  // groups whose instances were all skipped are dropped
  uint32_t partitionCount = 0;
  for(size_t k = 0; k < remap.size(); k++)
  {
    remap[k] = remap[k] ? partitionCount++ : ~0u;
  }

  for(size_t i = 0; i < m_instances.size(); i++)
  {
    m_instancePartitions[i] = remap[keys[i]];
  }

  return partitionCount;
}

uint32_t Scene::getPartitionBudget(const ScenePartitionConfig& partitionConfig) const
{
  uint32_t budget = partitionConfig.derivedMaxCount ? partitionConfig.derivedMaxCount : uint32_t(m_instances.size());
  if(partitionConfig.maxPartitions)
  {
    budget = std::min(budget, partitionConfig.maxPartitions);
  }
  return std::max(1u, budget);
}

// Top-down splits of the instance set, derived from meshoptimizer's spatial clusterizer
// (`meshopt_spatialClusterPoints` / `splitPoints` in `spatialorder.cpp`): one sorted order
// per axis, so a split is a linear partition of the other two; axis by widest extent; split
// aligned to the target size for uniform occupancy.
// `LAYOUT_BVH_SAH` instead picks the plane by a sah sweep over the instance bounds, which a
// center-only split cannot see - instances differ in size by orders of magnitude. The
// recursion is breadth first, so a level's nodes split in parallel over disjoint ranges.
uint32_t Scene::buildPartitionsBvh(const ScenePartitionConfig& partitionConfig)
{
  const uint32_t instanceCount = uint32_t(m_instances.size());
  const bool     useSah        = partitionConfig.layout == ScenePartitionConfig::LAYOUT_DERIVED_BVH_SAH;

  if(!instanceCount)
    return 0;

  std::vector<glm::vec3> centers(instanceCount);
  nvutils::parallel_batches<1024>(instanceCount, [&](uint64_t i) {
    centers[i] = (m_instances[i].bbox.lo + m_instances[i].bbox.hi) * 0.5f;
  });

  std::vector<uint32_t> axisOrder(size_t(instanceCount) * 3);
  nvutils::parallel_batches<1>(3, [&](uint64_t axisIndex) {
    const glm::length_t axis  = glm::length_t(axisIndex);
    uint32_t*           order = axisOrder.data() + axisIndex * instanceCount;
    std::iota(order, order + instanceCount, 0u);
    std::sort(order, order + instanceCount, [&](uint32_t a, uint32_t b) { return centers[a][axis] < centers[b][axis]; });
  });

  // per node scratch, indexed by the node's own range so the levels stay parallel
  std::vector<uint32_t> scratch(size_t(instanceCount) * 2);
  std::vector<uint8_t>  sides(instanceCount);
  std::vector<float>    sweepArea(useSah ? instanceCount : 0);

  const uint32_t maxLeaves       = getPartitionBudget(partitionConfig);
  const uint32_t targetInstances = std::max(1u, partitionConfig.derivedTargetInstances);

  auto surfaceArea = [](const shaderio::BBox& bbox) {
    glm::vec3 d = glm::max(bbox.hi - bbox.lo, glm::vec3(0.0f));
    return d.x * d.y + d.y * d.z + d.z * d.x;
  };
  const shaderio::BBox emptyBbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  // splits [begin,end) of all three orders in place, returns the split index or `begin` when
  // the node has no extent left to split along
  auto splitNode = [&](uint32_t begin, uint32_t end) {
    const uint32_t count = end - begin;

    uint32_t axis       = 0;
    float    axisExtent = -1.0f;
    for(uint32_t a = 0; a < 3; a++)
    {
      const uint32_t* order  = axisOrder.data() + size_t(a) * instanceCount;
      float           extent = centers[order[end - 1]][a] - centers[order[begin]][a];
      if(extent > axisExtent)
      {
        axisExtent = extent;
        axis       = a;
      }
    }
    if(axisExtent <= 0.0f)
      return begin;

    const uint32_t* order = axisOrder.data() + size_t(axis) * instanceCount;

    // meshopt splits at the middle, rounded up to a whole target size, so every partition
    // but the last holds exactly `targetInstances`
    uint32_t split = begin + std::max(1u, (count / 2 + targetInstances - 1) / targetInstances * targetInstances);
    split          = std::min(split, end - 1);

    if(useSah)
    {
      // the axis is already sorted, so the sah sweep is exact rather than binned: accumulate
      // the instance bounds from the right, then from the left, and keep the cheapest plane
      {
        shaderio::BBox bbox = emptyBbox;
        for(uint32_t i = end; i-- > begin;)
        {
          bbox.lo      = glm::min(bbox.lo, m_instances[order[i]].bbox.lo);
          bbox.hi      = glm::max(bbox.hi, m_instances[order[i]].bbox.hi);
          sweepArea[i] = surfaceArea(bbox);
        }
      }

      shaderio::BBox bbox     = emptyBbox;
      float          bestCost = FLT_MAX;
      for(uint32_t i = begin; i + 1 < end; i++)
      {
        bbox.lo = glm::min(bbox.lo, m_instances[order[i]].bbox.lo);
        bbox.hi = glm::max(bbox.hi, m_instances[order[i]].bbox.hi);

        float cost = surfaceArea(bbox) * float(i + 1 - begin) + sweepArea[i + 1] * float(end - i - 1);
        if(cost < bestCost)
        {
          bestCost = cost;
          split    = i + 1;
        }
      }
    }

    // mark the sides, then partition the other two orders while preserving their sort
    for(uint32_t i = begin; i < split; i++)
    {
      sides[order[i]] = 0;
    }
    for(uint32_t i = split; i < end; i++)
    {
      sides[order[i]] = 1;
    }

    uint32_t scratchSlot = 0;
    for(uint32_t a = 0; a < 3; a++)
    {
      if(a == axis)
        continue;

      uint32_t* other = axisOrder.data() + size_t(a) * instanceCount;
      uint32_t* temp  = scratch.data() + size_t(scratchSlot++) * instanceCount + begin;
      memcpy(temp, other + begin, sizeof(uint32_t) * count);

      uint32_t l = begin;
      uint32_t r = split;
      for(uint32_t i = 0; i < count; i++)
      {
        uint8_t side        = sides[temp[i]];
        other[side ? r : l] = temp[i];
        l += side ? 0 : 1;
        r += side ? 1 : 0;
      }
    }

    return split;
  };

  struct Node
  {
    uint32_t begin;
    uint32_t end;
  };

  std::vector<Node> leaves;
  std::vector<Node> level{{0, instanceCount}};

  while(!level.empty())
  {
    std::vector<Node> splittable;
    for(const Node& node : level)
    {
      if(node.end - node.begin > targetInstances)
        splittable.push_back(node);
      else
        leaves.push_back(node);
    }

    // a split adds exactly one partition, so what is left of the budget is how many of this
    // level's nodes may still split. Spend it on the fullest ones.
    size_t   pending = leaves.size() + splittable.size();
    uint32_t budget  = maxLeaves > pending ? maxLeaves - uint32_t(pending) : 0;
    if(budget < splittable.size())
    {
      std::sort(splittable.begin(), splittable.end(),
                [](const Node& a, const Node& b) { return (a.end - a.begin) > (b.end - b.begin); });
      leaves.insert(leaves.end(), splittable.begin() + budget, splittable.end());
      splittable.resize(budget);
    }

    std::vector<uint32_t> splits(splittable.size());
    nvutils::parallel_batches<1>(splittable.size(),
                                 [&](uint64_t n) { splits[n] = splitNode(splittable[n].begin, splittable[n].end); });

    std::vector<Node> next;
    next.reserve(splittable.size() * 2);
    for(size_t n = 0; n < splittable.size(); n++)
    {
      const Node& node = splittable[n];
      if(splits[n] <= node.begin || splits[n] >= node.end)
      {
        leaves.push_back(node);
        continue;
      }
      next.push_back({node.begin, splits[n]});
      next.push_back({splits[n], node.end});
    }
    level.swap(next);
  }

  // the orders are partitioned in place, so sorting the leaves by where their range starts
  // recovers the depth first order: partitions close in index are close in space, which is
  // what the grid numbers along a z curve for
  std::sort(leaves.begin(), leaves.end(), [](const Node& a, const Node& b) { return a.begin < b.begin; });

  const uint32_t* order = axisOrder.data();
  nvutils::parallel_batches<64>(leaves.size(), [&](uint64_t n) {
    for(uint32_t i = leaves[n].begin; i < leaves[n].end; i++)
    {
      m_instancePartitions[order[i]] = uint32_t(n);
    }
  });

  return uint32_t(leaves.size());
}

// Starts from the derived grid's initial cells and splits each into quadrants while it holds
// more than the target, so dense regions get finer cells where the derived grid would refine
// the whole scene. Every node is a cell of that grid refined by a power of two.
uint32_t Scene::buildPartitionsQuadtree(const ScenePartitionConfig& partitionConfig)
{
  const uint32_t instanceCount = uint32_t(m_instances.size());
  if(!instanceCount)
    return 0;

  const uint32_t perAxis         = getPartitionGridBaseCells(partitionConfig);
  const uint32_t maxLeaves       = getPartitionBudget(partitionConfig);
  const uint32_t targetInstances = std::max(1u, partitionConfig.derivedTargetInstances);
  // the leaf order packs the base cell and the path below it into 32 bits each, also stops
  // the descent of instances that share a position
  const uint32_t maxDepth = 16;

  m_partitionsPerAxis = perAxis;

  std::vector<glm::vec2> positions(instanceCount);
  nvutils::parallel_batches<1024>(instanceCount,
                                  [&](uint64_t i) { positions[i] = getPartitionGridPos(m_instances[i].bbox, perAxis); });

  struct Node
  {
    uint32_t begin;
    uint32_t end;
    // cell at `depth`, in a grid of `perAxis << depth` cells per axis
    uint32_t x;
    uint32_t y;
    uint32_t depth;
  };

  // counting sort into the base cells, each occupied one is a root
  std::vector<uint32_t> order(instanceCount);
  std::vector<Node>     heap;
  {
    std::vector<uint32_t> cellIndices(instanceCount);
    std::vector<uint32_t> cellOffsets(size_t(perAxis) * size_t(perAxis) + 1, 0);
    for(uint32_t i = 0; i < instanceCount; i++)
    {
      glm::uvec2 cell = glm::uvec2(positions[i]);
      cellIndices[i]  = cell.x + cell.y * perAxis;
      cellOffsets[cellIndices[i] + 1]++;
    }
    for(size_t c = 0; c + 1 < cellOffsets.size(); c++)
    {
      if(cellOffsets[c + 1])
      {
        uint32_t begin = cellOffsets[c];
        heap.push_back({begin, begin + cellOffsets[c + 1], uint32_t(c % perAxis), uint32_t(c / perAxis), 0});
      }
      cellOffsets[c + 1] += cellOffsets[c];
    }
    for(uint32_t i = 0; i < instanceCount; i++)
    {
      order[cellOffsets[cellIndices[i]]++] = i;
    }
  }

  // the fullest node splits first, across all depths, so a budget that runs out leaves
  // the remaining partitions as close to the target as it allows
  auto fuller = [](const Node& a, const Node& b) { return (a.end - a.begin) < (b.end - b.begin); };
  std::make_heap(heap.begin(), heap.end(), fuller);

  std::vector<uint32_t> scratch(instanceCount);
  std::vector<Node>     leaves;
  size_t                pending = heap.size();

  while(!heap.empty())
  {
    std::pop_heap(heap.begin(), heap.end(), fuller);
    Node node = heap.back();
    heap.pop_back();

    if(node.end - node.begin <= targetInstances)
    {
      // all others are no fuller
      leaves.push_back(node);
      leaves.insert(leaves.end(), heap.begin(), heap.end());
      break;
    }

    if(node.depth >= maxDepth)
    {
      leaves.push_back(node);
      continue;
    }

    // sort the range by quadrant, in z order to match the numbering below
    const int shift    = int(node.depth) + 1;
    auto      quadrant = [&](uint32_t i) {
      // scaling by a power of two is exact, so this agrees with the parent's cell
      uint32_t qx = uint32_t(std::floor(std::ldexp(positions[i].x, shift))) - node.x * 2;
      uint32_t qy = uint32_t(std::floor(std::ldexp(positions[i].y, shift))) - node.y * 2;
      return std::min(qx, 1u) + std::min(qy, 1u) * 2;
    };

    glm::uvec4 counts(0);
    for(uint32_t i = node.begin; i < node.end; i++)
    {
      counts[quadrant(order[i])]++;
    }

    // a split adds a partition per occupied quadrant beyond the first, a node that does not
    // fit the budget stays a leaf while smaller splits may still fit
    uint32_t occupied = (counts.x ? 1 : 0) + (counts.y ? 1 : 0) + (counts.z ? 1 : 0) + (counts.w ? 1 : 0);
    if(pending + occupied - 1 > maxLeaves)
    {
      leaves.push_back(node);
      continue;
    }
    pending += occupied - 1;

    glm::uvec4 offsets(node.begin, node.begin + counts.x, node.begin + counts.x + counts.y,
                       node.begin + counts.x + counts.y + counts.z);
    for(uint32_t i = node.begin; i < node.end; i++)
    {
      scratch[offsets[quadrant(order[i])]++] = order[i];
    }
    memcpy(order.data() + node.begin, scratch.data() + node.begin, sizeof(uint32_t) * (node.end - node.begin));

    uint32_t begin = node.begin;
    for(uint32_t q = 0; q < 4; q++)
    {
      if(counts[q])
      {
        heap.push_back({begin, begin + counts[q], node.x * 2 + (q & 1), node.y * 2 + (q >> 1), node.depth + 1});
        std::push_heap(heap.begin(), heap.end(), fuller);
      }
      begin += counts[q];
    }
  }

  // z order of the base cell, then of the path below it aligned to `maxDepth`, which is the
  // depth first order of the quadrants. Partitions close in index are close in space, as for
  // the grid.
  std::vector<std::pair<uint64_t, uint32_t>> leafKeys(leaves.size());
  for(size_t n = 0; n < leaves.size(); n++)
  {
    const Node& node = leaves[n];
    uint32_t    mask = (1u << node.depth) - 1;
    uint64_t    base = part1By1(node.x >> node.depth) | (part1By1(node.y >> node.depth) << 1);
    uint64_t below = uint64_t(part1By1(node.x & mask) | (part1By1(node.y & mask) << 1)) << ((maxDepth - node.depth) * 2);
    leafKeys[n] = {(base << 32) | below, uint32_t(n)};
  }
  std::sort(leafKeys.begin(), leafKeys.end());

  nvutils::parallel_batches<64>(leafKeys.size(), [&](uint64_t n) {
    const Node& node = leaves[leafKeys[n].second];
    for(uint32_t i = node.begin; i < node.end; i++)
    {
      m_instancePartitions[order[i]] = uint32_t(n);
    }
  });

  return uint32_t(leaves.size());
}

void Scene::updatePartitions(const ScenePartitionConfig& partitionConfig)
{
  // the assignment only depends on the instances and the config, both rarely change,
  // while the renderer is re-created on every config toggle
  if(m_partitionConfig == partitionConfig && m_instancePartitions.size() == m_instances.size()
     && m_partitionSceneBbox.lo == m_gridBbox.lo && m_partitionSceneBbox.hi == m_gridBbox.hi)
  {
    return;
  }

  m_partitionConfig = partitionConfig;
  // `m_bbox` is the model, only `m_gridBbox` covers the instances of all grid copies
  m_partitionSceneBbox = m_gridBbox;
  m_instancePartitions.resize(m_instances.size());
  m_partitionsPerAxis = 0;

  uint32_t partitionCount = 0;
  switch(partitionConfig.layout)
  {
    case ScenePartitionConfig::LAYOUT_FIXED_GRID:
      partitionCount = partitionConfig.gridCellSize > 0 ? buildPartitionsCells(partitionConfig) :
                                                          buildPartitionsGrid(partitionConfig);
      break;
    case ScenePartitionConfig::LAYOUT_DERIVED_BVH_MEDIAN:
    case ScenePartitionConfig::LAYOUT_DERIVED_BVH_SAH:
      partitionCount = buildPartitionsBvh(partitionConfig);
      break;
    case ScenePartitionConfig::LAYOUT_DERIVED_QUADTREE:
      partitionCount = buildPartitionsQuadtree(partitionConfig);
      break;
    case ScenePartitionConfig::LAYOUT_NODE_REGEX:
      if(m_instanceNodeGroups.empty())
      {
        LOGW("scene partitions: the node regex layout has no node groups, see --partitionnodes\n");
      }
      partitionCount = buildPartitionsNodeGroups();
      break;
    default:
      partitionCount = buildPartitionsGrid(partitionConfig);
      break;
  }

  m_partitions.assign(partitionCount, {{{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0}, 0, 0});

  for(size_t i = 0; i < m_instances.size(); i++)
  {
    Partition& partition = m_partitions[m_instancePartitions[i]];

    partition.bbox.lo = glm::min(partition.bbox.lo, m_instances[i].bbox.lo);
    partition.bbox.hi = glm::max(partition.bbox.hi, m_instances[i].bbox.hi);
    partition.instanceCount++;
  }

  // prefix sum the counts, then scatter instance ids so each partition owns a contiguous range
  {
    uint32_t offset = 0;
    for(Partition& partition : m_partitions)
    {
      partition.instanceOffset = offset;
      offset += partition.instanceCount;
    }
    m_partitionInstanceOrder.resize(m_instances.size());
    std::vector<uint32_t> fill(m_partitions.size(), 0);
    for(size_t i = 0; i < m_instances.size(); i++)
    {
      uint32_t partitionIndex = m_instancePartitions[i];
      m_partitionInstanceOrder[m_partitions[partitionIndex].instanceOffset + fill[partitionIndex]++] = uint32_t(i);
    }
  }

  uint32_t aboveTarget  = 0;
  uint32_t maxInstances = 0;
  for(const Partition& partition : m_partitions)
  {
    aboveTarget += partition.instanceCount > partitionConfig.derivedMaxInstances ? 1 : 0;
    maxInstances = std::max(maxInstances, partition.instanceCount);
  }

  m_partitionHistogram.assign(size_t(std::min(maxInstances, PARTITION_HISTOGRAM_MAX)) + 1, 0);
  for(const Partition& partition : m_partitions)
  {
    m_partitionHistogram[std::min(partition.instanceCount, PARTITION_HISTOGRAM_MAX)]++;
  }
  m_partitionHistogramMax = *std::max_element(m_partitionHistogram.begin(), m_partitionHistogram.end());

  const char* layoutNames[ScenePartitionConfig::LAYOUT_COUNT] = {
      "fixed grid", "derived grid", "derived bvh median", "derived bvh sah", "derived quadtree", "node regex"};

  std::string layout =
      layoutNames[partitionConfig.layout < ScenePartitionConfig::LAYOUT_COUNT ? partitionConfig.layout : uint32_t(ScenePartitionConfig::LAYOUT_DERIVED_GRID)];
  if(m_partitionsPerAxis)
  {
    layout += fmt::format(" {}x{}", m_partitionsPerAxis, m_partitionsPerAxis);
  }
  else if(partitionConfig.layout == ScenePartitionConfig::LAYOUT_FIXED_GRID && partitionConfig.gridCellSize > 0)
  {
    layout += fmt::format(" {} units", partitionConfig.gridCellSize);
  }

  // only the derived grid refines towards `derivedMaxInstances`
  std::string aboveMax;
  if(partitionConfig.layout == ScenePartitionConfig::LAYOUT_DERIVED_GRID)
  {
    aboveMax = fmt::format(", {} above the {} max", aboveTarget, partitionConfig.derivedMaxInstances);
  }
  LOGI("scene partitions: %s, %zu used, %d instances max%s\n", layout.c_str(), m_partitions.size(), maxInstances,
       aboveMax.c_str());
  LOGI("scene partitions: bounds %.3f %.3f %.3f to %.3f %.3f %.3f\n", m_gridBbox.lo.x, m_gridBbox.lo.y, m_gridBbox.lo.z,
       m_gridBbox.hi.x, m_gridBbox.hi.y, m_gridBbox.hi.z);
  if(partitionConfig.maxPartitions && m_partitions.size() > partitionConfig.maxPartitions)
  {
    LOGW("scene partitions: %zu used exceed the device limit of %d\n", m_partitions.size(), partitionConfig.maxPartitions);
  }
}

}  // namespace lodclusters
