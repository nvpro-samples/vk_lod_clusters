/**
 * clusterlod - a small "library"/example built on top of meshoptimizer to generate cluster LOD hierarchies
 * This is intended to either be used as is, or as a reference for implementing similar functionality in your engine.
 *
 * To use this code, you need to have one source file which includes meshoptimizer.h and defines CLUSTERLOD_IMPLEMENTATION
 * before including this file. Other source files in your project can just include this file and use the provided functions.
 *
 * Copyright (C) 2016-2025, by Arseny Kapoulkine (arseny.kapoulkine@gmail.com)
 * This code is distributed under the MIT License. See notice at the end of this file.
 */
// clang-format off

// This file originated from
// https://github.com/zeux/meshoptimizer/blob/67b720df097d3f8ccc2318d7fe80a8b9018c3dc6/demo/clusterlod.h
// and was modified by NVIDIA CORPORATION
//
// - multi-threading support through the `clodIteration` callback and `clodBuild_iterationTask`

#pragma once

#include <stddef.h>

struct clodConfig
{
	// configuration of each cluster; maps to meshopt_buildMeshlets* parameters
	size_t max_vertices;
	size_t min_triangles;
	size_t max_triangles;

	// partitioning setup; maps to meshopt_partitionClusters parameters
	// note: partition size is the target size, not maximum; actual partitions may be up to 1/3 larger (e.g. target 24 results in maximum 32)
	bool partition_spatial;
	size_t partition_size;

	// clusterization setup; maps to meshopt_buildMeshletsSpatial / meshopt_buildMeshletsFlex
	bool cluster_spatial;
	float cluster_fill_weight;
	float cluster_split_factor;

	// every level aims to reduce the number of triangles by ratio, and considers clusters that don't reach the threshold stuck
	float simplify_ratio;
	float simplify_threshold;

	// to compute the error of simplified clusters, we use the formula that combines previous accumulated error as follows:
	// max(previous_error * simplify_error_merge_previous, current_error) + current_error * simplify_error_merge_additive
	float simplify_error_merge_previous;
	float simplify_error_merge_additive;

	// amplify the error of clusters that go through sloppy simplification to account for appearance degradation
	float simplify_error_factor_sloppy;

	// use permissive simplification instead of regular simplification (make sure to use attribute_protect_mask if this is set!)
	bool simplify_permissive;

	// use permissive or sloppy simplification but only if regular simplification gets stuck
	bool simplify_fallback_permissive;
	bool simplify_fallback_sloppy;

	// should clodCluster::bounds be computed based on the geometry of each cluster
	bool optimize_bounds;

	// should clodCluster::indices be optimized for rasterization
	bool optimize_raster;
};

struct clodMesh
{
	// input triangle indices
	const unsigned int* indices;
	size_t index_count;

	// total vertex count
	size_t vertex_count;

	// input vertex positions; must be 3 floats per vertex
	const float* vertex_positions;
	size_t vertex_positions_stride;

	// input vertex attributes; used for attribute-aware simplification and permissive simplification
	const float* vertex_attributes;
	size_t vertex_attributes_stride;

	// input vertex locks; allows to preserve additional seams (when not using attribute_protect_mask) or lock vertices via meshopt_SimplifyVertex_* flags
	const unsigned char* vertex_lock;

	// attribute weights for attribute-aware simplification; maps to meshopt_simplifyWithAttributes parameters
	const float* attribute_weights;
	size_t attribute_count;

	// attribute mask to flag attribute discontinuities for permissive simplification; mask (1<<K) corresponds to attribute K
	unsigned int attribute_protect_mask;
};

// To compute approximate (perspective) projection error of a cluster in screen space (0..1; multiply by screen height to get pixels):
// - camera_proj is projection[1][1], or cot(fovy/2); camera_znear is *positive* near plane distance
// - for simplicity, we ignore perspective distortion and use rotationally invariant projection size estimation
// - return: bounds.error / max(distance(bounds.center, camera_position) - bounds.radius, camera_znear) * (camera_proj * 0.5f)
struct clodBounds
{
	// sphere bounds, in mesh coordinate space
	float center[3];
	float radius;

	// combined simplification error, in mesh coordinate space
	float error;
};

struct clodCluster
{
	// index of more refined group (with more triangles) that produced this cluster during simplification, or -1 for original geometry
	int refined;

	// cluster bounds; should only be used for culling, as bounds.error is not monotonic across DAG
	clodBounds bounds;

	// cluster indices; refer to the original mesh vertex buffer
	const unsigned int* indices;
	size_t index_count;

	// cluster vertex count; indices[] has vertex_count unique entries
	// can be used to extract local index buffer from indices[] using meshopt_buildMeshletsScan
	size_t vertex_count;
};

struct clodGroup
{
	// DAG level the group was generated at
	int depth;

	// simplified group bounds (reflects error for clusters with clodCluster::refined == group id; error is FLT_MAX for terminal groups)
	// cluster should be rendered if:
	// 1. clodGroup::simplified for the group it's in is over error threshold
	// 2. cluster.refined is -1 *or* clodGroup::simplified for groups[cluster.refined].simplified is at or under error threshold
	clodBounds simplified;
};

// gets called for each group
// returned value gets saved for clusters emitted from this group (clodCluster::refined)
typedef int (*clodOutput)(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count);

// gets called for each lod level iteration, except last, which directly calls clodOutput.
// user must call `clodBuild_iterationTask` passing through `intermediate_context` and with task_index from 0 to task_count-1.
typedef void (*clodIteration)(void* iteration_context, void* output_context, int depth, size_t task_count);


#ifdef __cplusplus
extern "C"
{
#endif

// default configuration optimized for rasterization / raytracing
clodConfig clodDefaultConfig(size_t max_triangles);
clodConfig clodDefaultConfigRT(size_t max_triangles);

// build cluster LOD hierarchy, calling output callbacks as new clusters and groups are generated
// returns the total number of clusters produced
size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback);

// If `iteration_callback` is used, it must pass through the `iteration_context` unaltered
// and call this function `task_count` many times.
// The provided `output_context` is passed to `clodOutput`.
// Is thread-safe as long as the output callback is thread-safe or null.
void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t task_index);

#ifdef __cplusplus
} // extern "C"

template <typename Output>
size_t clodBuild(clodConfig config, clodMesh mesh, Output output)
{
	struct Call
	{
		static int output(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count)
		{
			return (*static_cast<Output*>(output_context))(group, clusters, cluster_count);
		}
	};

	return clodBuild(config, mesh, &output, &Call::output, nullptr);
}
#endif

#ifdef CLUSTERLOD_IMPLEMENTATION
// For reference, see the original Nanite paper:
// Brian Karis. Nanite: A Deep Dive. 2021
#include <float.h>
#include <math.h>
#include <string.h>

#include <algorithm>
#include <vector>
#include <atomic>

namespace clod
{

struct Cluster
{
	size_t vertices;
	std::vector<unsigned int> indices;

	int group;
	int refined;

	clodBounds bounds;
};

static clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error)
{
	meshopt_Bounds bounds = meshopt_computeClusterBounds(&indices[0], indices.size(), mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);

	clodBounds result;
	result.center[0] = bounds.center[0];
	result.center[1] = bounds.center[1];
	result.center[2] = bounds.center[2];
	result.radius = bounds.radius;
	result.error = error;
	return result;
}

static clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group)
{
	std::vector<clodBounds> bounds(group.size());
	for (size_t j = 0; j < group.size(); ++j)
		bounds[j] = clusters[group[j]].bounds;

	meshopt_Bounds merged = meshopt_computeSphereBounds(&bounds[0].center[0], bounds.size(), sizeof(clodBounds), &bounds[0].radius, sizeof(clodBounds));

	clodBounds result = {};
	result.center[0] = merged.center[0];
	result.center[1] = merged.center[1];
	result.center[2] = merged.center[2];
	result.radius = merged.radius;

	// merged bounds error must be conservative wrt cluster errors
	result.error = 0.f;
	for (size_t j = 0; j < group.size(); ++j)
		result.error = std::max(result.error, clusters[group[j]].bounds.error);

	return result;
}

static std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count)
{
	size_t max_meshlets = meshopt_buildMeshletsBound(index_count, config.max_vertices, config.min_triangles);

	std::vector<meshopt_Meshlet> meshlets(max_meshlets);
	std::vector<unsigned int> meshlet_vertices(index_count);

#if MESHOPTIMIZER_VERSION < 1000
	std::vector<unsigned char> meshlet_triangles(index_count + max_meshlets * 3); // account for 4b alignment
#else
	std::vector<unsigned char> meshlet_triangles(index_count);
#endif

	if (config.cluster_spatial)
		meshlets.resize(meshopt_buildMeshletsSpatial(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, config.cluster_fill_weight));
	else
		meshlets.resize(meshopt_buildMeshletsFlex(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, 0.f, config.cluster_split_factor));

	std::vector<Cluster> clusters(meshlets.size());

	for (size_t i = 0; i < meshlets.size(); ++i)
	{
		const meshopt_Meshlet& meshlet = meshlets[i];

		if (config.optimize_raster)
			meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, meshlet.vertex_count);

		clusters[i].vertices = meshlet.vertex_count;

		// note: we discard meshlet-local indices; they can be recovered by the caller using meshopt_buildMeshletsScan
		clusters[i].indices.resize(meshlet.triangle_count * 3);
		for (size_t j = 0; j < meshlet.triangle_count * 3; ++j)
			clusters[i].indices[j] = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]];

		clusters[i].group = -1;
		clusters[i].refined = -1;
	}

	return clusters;
}

static std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap)
{
	if (pending.size() <= config.partition_size)
		return {pending};

	std::vector<unsigned int> cluster_indices;
	std::vector<unsigned int> cluster_counts(pending.size());

	size_t total_index_count = 0;
	for (size_t i = 0; i < pending.size(); ++i)
		total_index_count += clusters[pending[i]].indices.size();

	cluster_indices.reserve(total_index_count);

	for (size_t i = 0; i < pending.size(); ++i)
	{
		const Cluster& cluster = clusters[pending[i]];

		cluster_counts[i] = unsigned(cluster.indices.size());

		for (size_t j = 0; j < cluster.indices.size(); ++j)
			cluster_indices.push_back(remap[cluster.indices[j]]);
	}

	std::vector<unsigned int> cluster_part(pending.size());
	size_t partition_count = meshopt_partitionClusters(&cluster_part[0], &cluster_indices[0], cluster_indices.size(), &cluster_counts[0], cluster_counts.size(),
	    config.partition_spatial ? mesh.vertex_positions : NULL, remap.size(), mesh.vertex_positions_stride, config.partition_size);

	std::vector<std::vector<int> > partitions(partition_count);
	for (size_t i = 0; i < partition_count; ++i)
		partitions[i].reserve(config.partition_size + config.partition_size / 3);

	for (size_t i = 0; i < pending.size(); ++i)
		partitions[cluster_part[i]].push_back(pending[i]);

	return partitions;
}

static void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock)
{
	// for each remapped vertex, use bit 7 as temporary storage to indicate that the vertex has been used by a different group previously
	for (size_t i = 0; i < locks.size(); ++i)
		locks[i] &= ~((1 << 0) | (1 << 7));

	for (size_t i = 0; i < groups.size(); ++i)
	{
		// mark all remapped vertices as locked if seen by a prior group
		for (size_t j = 0; j < groups[i].size(); ++j)
		{
			const Cluster& cluster = clusters[groups[i][j]];

			for (size_t k = 0; k < cluster.indices.size(); ++k)
			{
				unsigned int v = cluster.indices[k];
				unsigned int r = remap[v];

				locks[r] |= locks[r] >> 7;
			}
		}

		// mark all remapped vertices as seen
		for (size_t j = 0; j < groups[i].size(); ++j)
		{
			const Cluster& cluster = clusters[groups[i][j]];

			for (size_t k = 0; k < cluster.indices.size(); ++k)
			{
				unsigned int v = cluster.indices[k];
				unsigned int r = remap[v];

				locks[r] |= 1 << 7;
			}
		}
	}

	for (size_t i = 0; i < locks.size(); ++i)
	{
		unsigned int r = remap[i];

		// consistently lock all vertices with the same position; keep protect bit if set
		locks[i] = (locks[r] & 1) | (locks[i] & meshopt_SimplifyVertex_Protect);

		if (vertex_lock)
			locks[i] |= vertex_lock[i];
	}
}

struct SloppyVertex
{
	float x, y, z;
	unsigned int id;
};

static void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	std::vector<SloppyVertex> subset(indices.size());
	std::vector<unsigned char> subset_locks(indices.size());

	lod.resize(indices.size());

	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);

	// deindex the mesh subset to avoid calling simplifySloppy on the entire vertex buffer (which is prohibitively expensive without sparsity)
	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		assert(v < mesh.vertex_count);

		subset[i].x = mesh.vertex_positions[v * positions_stride + 0];
		subset[i].y = mesh.vertex_positions[v * positions_stride + 1];
		subset[i].z = mesh.vertex_positions[v * positions_stride + 2];
		subset[i].id = v;

		subset_locks[i] = locks[v];
		lod[i] = unsigned(i);
	}

	lod.resize(meshopt_simplifySloppy(&lod[0], &lod[0], lod.size(), &subset[0].x, subset.size(), sizeof(SloppyVertex), subset_locks.data(), target_count, FLT_MAX, error));

	// convert error to absolute
	*error *= meshopt_simplifyScale(&subset[0].x, subset.size(), sizeof(SloppyVertex));

	// restore original vertex indices
	for (size_t i = 0; i < lod.size(); ++i)
		lod[i] = subset[lod[i]].id;
}

static std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	if (target_count > indices.size())
		return indices;

	std::vector<unsigned int> lod(indices.size());

	unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | (config.simplify_permissive ? meshopt_SimplifyPermissive : 0);

	lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
	    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
	    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
	    &locks[0], target_count, FLT_MAX, options, error));

	if (lod.size() > target_count && config.simplify_fallback_permissive && !config.simplify_permissive)
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
		    &locks[0], target_count, FLT_MAX, options | meshopt_SimplifyPermissive, error));

	// while it's possible to call simplifySloppy directly, it doesn't support sparsity or absolute error, so we need to do some extra work
	if (lod.size() > target_count && config.simplify_fallback_sloppy)
	{
		simplifyFallback(lod, mesh, indices, locks, target_count, error);
		*error *= config.simplify_error_factor_sloppy; // scale error up to account for appearance degradation
	}

	return lod;
}

static int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback)
{
	std::vector<clodCluster> group_clusters(group.size());

	for (size_t i = 0; i < group.size(); ++i)
	{
		const Cluster& cluster = clusters[group[i]];
		clodCluster& result = group_clusters[i];

		result.refined = cluster.refined;
		result.bounds = (config.optimize_bounds && cluster.refined != -1) ? boundsCompute(mesh, cluster.indices, cluster.bounds.error) : cluster.bounds;
		result.indices = cluster.indices.data();
		result.index_count = cluster.indices.size();
		result.vertex_count = cluster.vertices;
	}

	return output_callback ? output_callback(output_context, {depth, simplified}, group_clusters.data(), group_clusters.size()) : -1;
}

struct IterationContext
{
	clodConfig config;
	clodMesh   mesh;
	clodOutput output_callback = nullptr;

	// persistent
	std::vector<unsigned char> locks;
	std::vector<unsigned int>  remap;

	int depth = 0;

	// grows over all iterations
	std::vector<Cluster> clusters;
	std::atomic<size_t>  next_cluster = {}; // lock free allocation index into above

	// reset every iteration
	std::vector<std::vector<int>> groups;

	std::vector<int>    pending;
	std::atomic<size_t> next_pending = {}; // lock free allocation index into above
};

} // namespace clod

clodConfig clodDefaultConfig(size_t max_triangles)
{
	assert(max_triangles >= 4 && max_triangles <= 256);

	clodConfig config = {};
	config.max_vertices = max_triangles;
	config.min_triangles = max_triangles / 3;
	config.max_triangles = max_triangles;

#if MESHOPTIMIZER_VERSION < 1000
	config.min_triangles &= ~3; // account for 4b alignment
#endif

	config.partition_spatial = true;
	config.partition_size = 16;

	config.cluster_spatial = false;
	config.cluster_split_factor = 2.0f;

	config.optimize_raster = true;

	config.simplify_ratio = 0.5f;
	config.simplify_threshold = 0.85f;
	config.simplify_error_merge_previous = 1.0f;
	config.simplify_error_factor_sloppy = 2.0f;
	config.simplify_permissive = true;
	config.simplify_fallback_permissive = false; // note: by default we run in permissive mode, but it's also possible to disable that and use it only as a fallback
	config.simplify_fallback_sloppy = true;

	return config;
}

clodConfig clodDefaultConfigRT(size_t max_triangles)
{
	clodConfig config = clodDefaultConfig(max_triangles);

	config.max_vertices = std::max(size_t(256), max_triangles + max_triangles / 2);

	config.cluster_spatial = true;
	config.cluster_fill_weight = 0.5f;

	config.optimize_raster = false;

	return config;
}

void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t i)
{
	using namespace clod;

	IterationContext&              context  = *(IterationContext*)iteration_context;
	std::vector<std::vector<int>>& groups   = context.groups;
	std::vector<Cluster>&          clusters = context.clusters;
	std::vector<unsigned char>&    locks    = context.locks;
	const clodMesh&                mesh     = context.mesh;
	const clodConfig&              config   = context.config;
	int                            depth    = context.depth;

	std::vector<unsigned int> merged;
	merged.reserve(groups[i].size() * config.max_triangles * 3);
	for (size_t j = 0; j < groups[i].size(); ++j)
		merged.insert(merged.end(), clusters[groups[i][j]].indices.begin(), clusters[groups[i][j]].indices.end());

	size_t target_size = size_t((merged.size() / 3) * config.simplify_ratio) * 3;

	// enforce bounds and error monotonicity
	// note: it is incorrect to use the precise bounds of the merged or simplified mesh, because this may violate monotonicity
	clodBounds bounds = boundsMerge(clusters, groups[i]);

	float error = 0.f;
	std::vector<unsigned int> simplified = simplify(config, mesh, merged, locks, target_size, &error);
	if (simplified.size() > merged.size() * config.simplify_threshold)
	{
		bounds.error = FLT_MAX; // terminal group, won't simplify further
		outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback);
		return; // simplification is stuck; abandon the merge
	}

	// enforce error monotonicity (with an optional hierarchical factor to separate transitions more)
	bounds.error = std::max(bounds.error * config.simplify_error_merge_previous, error) + error * config.simplify_error_merge_additive;

	// output the new group with all clusters; the resulting id will be recorded in new clusters as clodCluster::refined
	int refined = outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback);

	// discard clusters from the group - they won't be used anymore
	for (size_t j = 0; j < groups[i].size(); ++j)
		clusters[groups[i][j]].indices = std::vector<unsigned int>();

	std::vector<Cluster> split = clusterize(config, mesh, simplified.data(), simplified.size());

	size_t cluster_index = context.next_cluster.fetch_add(split.size());
	size_t pending_index = context.next_pending.fetch_add(split.size());

	for (Cluster& cluster : split)
	{
		cluster.refined = refined;

		// update cluster group bounds to the group-merged bounds; this ensures that we compute the group bounds for whatever group this cluster will be part of conservatively
		cluster.bounds = bounds;

		// enqueue new cluster for further processing
		context.pending[pending_index++]  = int(cluster_index);
		context.clusters[cluster_index++] = std::move(cluster);
	}
}

size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback)
{
	using namespace clod;

	assert(mesh.vertex_attributes_stride % sizeof(float) == 0);
	assert(mesh.attribute_count * sizeof(float) <= mesh.vertex_attributes_stride);
	assert(mesh.attribute_protect_mask < (1u << (mesh.vertex_attributes_stride / sizeof(float))));

	IterationContext context;
	context.config = config;
	context.mesh = mesh;
	context.output_callback = output_callback;
	context.locks.resize(mesh.vertex_count);
	context.remap.resize(mesh.vertex_count);

	// for cluster connectivity, we need a position-only remap that maps vertices with the same position to the same index
	meshopt_generatePositionRemap(&context.remap[0], mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);

	// set up protect bits on UV seams for permissive mode
	if (mesh.attribute_protect_mask)
	{
		size_t max_attributes = mesh.vertex_attributes_stride / sizeof(float);

		for (size_t i = 0; i < mesh.vertex_count; ++i)
		{
			unsigned int r = context.remap[i]; // canonical vertex with the same position

			for (size_t j = 0; j < max_attributes; ++j)
				if (r != i && (mesh.attribute_protect_mask & (1u << j)) && mesh.vertex_attributes[i * max_attributes + j] != mesh.vertex_attributes[r * max_attributes + j])
					context.locks[i] |= meshopt_SimplifyVertex_Protect;
		}
	}

	// initial clusterization splits the original mesh
	context.clusters = clusterize(config, mesh, mesh.indices, mesh.index_count);
	context.next_cluster = context.clusters.size();

	// compute initial precise bounds; subsequent bounds will be using group-merged bounds
	for (Cluster& cluster : context.clusters)
		cluster.bounds = boundsCompute(mesh, cluster.indices, 0.f);

	context.pending.resize(context.clusters.size());
	for (size_t i = 0; i < context.clusters.size(); ++i)
		context.pending[i] = int(i);



	// merge and simplify clusters until we can't merge anymore
	while (context.pending.size() > 1)
	{
		context.groups = partition(config, mesh, context.clusters, context.pending, context.remap);

		// lock-free allocation assumes we will not create more new clusters than pending
		context.clusters.resize(context.clusters.size() + context.pending.size());
		context.next_pending = 0;

		// mark boundaries between groups with a lock bit to avoid gaps in simplified result
		lockBoundary(context.locks, context.groups, context.clusters, context.remap, context.mesh.vertex_lock);

		// every group needs to be simplified now
		if (iteration_callback)
		{
			iteration_callback(&context, output_context, context.depth, context.groups.size());
		}
		else
		{
			for (size_t i = 0; i < context.groups.size(); ++i)
			{
				clodBuild_iterationTask(&context, output_context, i);
			}
		}

		// adjust sizes of arrays to actual usage
		context.pending.resize(context.next_pending);
		context.clusters.resize(context.next_cluster);

		context.depth++;
	}

	if (context.pending.size())
	{
		assert(context.pending.size() == 1);
		const Cluster& cluster = context.clusters[context.pending[0]];

		clodBounds bounds = cluster.bounds;
		bounds.error = FLT_MAX; // terminal group, won't simplify further

		outputGroup(config, mesh, context.clusters, context.pending, bounds, context.depth, output_context, output_callback);
	}

	return context.clusters.size();
}
#endif

/**
 * Copyright (c) 2016-2025 Arseny Kapoulkine
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
// clang-format on
