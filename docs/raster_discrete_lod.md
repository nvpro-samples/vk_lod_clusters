# Discrete LoD for rasterization with cluster-based continuous level of detail

This technique is the rasterization counterpart to ["BLAS Caching"](blas_caching.md) and reuses some of its logistics.

Instances that are far enough away to get by with a *single*, fully resident, discrete level of detail (LoD) level skip
the per-cluster LoD metric entirely and render that level as a whole.

## Motivation

Ray tracing has a strong reason to render a discrete LoD level: a BLAS is built per instance, and a BLAS built from one
discrete level is rotation invariant, so it can be cached across frames and shared between instances. Rasterization,
however doesn't really need this and can do the most accurate DAG cut per-instance.

The reason to do it anyway is a **hybrid renderer**, where the same scene is both rasterized and ray traced, for example
rasterizing primary visibility and tracing secondary rays. There the two paths have to agree on what geometry is
resident, and that is much easier when they want the *same* geometry:

* BLAS caching pins one (can be extended to multiple) discrete LoD level of a geometry in memory for as long as any instance uses the cached BLAS.
  If rasterization meanwhile walks the LoD DAG for a continuous cut, it pulls in a different, finer set of clusters causing divergence
  between the two and is most likely causing self-shadow or self-reflection artifacts.
* If rasterization instead renders from the same discrete levels, the clusters it uses are the ones the cached BLAS
  already keeps resident.  The result is we don't have divergence between the two anymore.

This is reflected directly in the data. `Geometry::discreteLodLevel` (see [`shaderio_scene.h`](../shaders/shaderio_scene.h))
is one field used by both paths, the level the cached BLAS was built for in ray tracing and the level from which on
everything is fully resident in rasterization. The streaming age filter that keeps those levels alive is the same code
for both, see `streamingAgeFilter` in [`streaming.glsl`](../shaders/streaming.glsl).

There is another benefit: we can reduce traversal work a little bit, by only following one LoD level for distant
instances. While this isn't a huge gain it's still there.

This is safe because each LoD level is a complete decimation of the whole mesh, the same property the cached BLAS
relies on. Rendering all clusters of the instance's `lodLevelMin` is therefore watertight and has equal or higher detail
everywhere than its continuous cut, at the cost of potentially more triangles.

## Algorithm

### Eligibility

Classified per instance, an instance takes the discrete path when all of these are met:

1. It survives the existing instance frustum and occlusion culling.
2. `lodLevelMin` is within the last `--discreteenabledlevels` levels of the geometry. As with BLAS caching, counting
   from the tail rather than the front keeps the geometric complexity predictable across geometries with different
   level counts.
3. The level is fully resident, `lodLevelMin >= geometry.discreteLodLevel`. Without streaming this is always true.
4. The instance's continuous cut fits `--discretelodrange`, so `lodLevelMax - lodLevelMin < range`. At `1` only
   instances that already resolve to a single level qualify and the technique emits exactly the clusters the traversal
   would have, larger values trade traversal savings against extra triangles.

We still have a shortcut for directly skipping the instance traversal for the lowest-detail use-case that goes directly
to a single cluster.

The LoD range needs `lodLevelMax`, which comes from the same per-level minimum sphere test
(`LodLevel::minBoundingSphereRadius` / `minMaxQuadricError`) that "BLAS sharing" already uses. Both share
`classifyInstanceLod` in [`traversal.glsl`](../shaders/traversal.glsl).

### Seeding the traversal

An eligible instance does not get its own kernel. It enqueues the LoD level's node, that is
`geometry.nodes[rootChildOffset + lodLevel]`, instead of the geometry's root node, and tags it with
`TRAVERSAL_DISCRETE_BIT` in `TraversalInfo::instanceID` (instance IDs use 31 bits, the same packing trick "BLAS sharing"
uses). From there:

* the node traversal always descends a tagged node, no metric is evaluated;
* the group traversal treats a tagged group as if its clusters had no generating group, which is the existing
  "always draw" path;
* a LoD level whose node is already a leaf goes straight into the group queue.

The point of seeding a node rather than iterating the groups directly is that there is a benefit in using the **existing hierarchical culling**.

We tried a dedicated kernel with flat iteration over groups and clusters, but it was worse. Measured as
"Traversal Run" GPU time:

| | Caldera | Zorah |
| --- | --- | --- |
| discrete lod off | 115 - 119 us | 124 - 127 us |
| flat composition | 226 - 252 us | 189 - 192 us |
| seeded traversal | 111 - 118 us | 116 - 122 us |

### Keeping the levels resident

An instance on the discrete path only ever touches the groups of its own level, so the coarser levels it relies on
would age out and break the geometry's residency guarantee. The classification therefore does an `atomicMin` of the
used level into `geometryCachedInfos[geometryID].cachedLevel`, and the streaming age filter resets the age of every
resident group at or above that level. This is exactly the mechanism "BLAS caching" uses, the host simply enables it for
either technique.

On the host, `SceneStreaming::handleDiscreteLod` recomputes, for every geometry touched by a load or unload, the level
from which on all levels are fully loaded, and publishes it as a geometry patch. Because that patch is built from the
same bookkeeping as the load and unload it travels with, and because scene patches are applied before the traversal,
the device never sees a `discreteLodLevel` that disagrees with the group addresses.

## Results and limitations

The technique is off by default (`--discretelod`) as this sample doesn't yet implement a hybrid renderer.

The *"discrete lod"* visualization (`--visualize 11`) colors each instance by how it built its cluster list:

* **green**: a single discrete LoD level, including the lowest detail shortcut
* **red**: the regular LoD hierarchy traversal

## Implementation

In the source look for `USE_DISCRETE_LOD` and `useDiscreteLod`.

Key changes in the shaders and device code are:
* [`traversal_init_discrete_lod.comp.glsl`](../shaders/traversal_init_discrete_lod.comp.glsl): replaces
  `traversal_init.comp.glsl` when the technique is active. Classifies the instance, seeds the traversal at the LoD
  level's node and does the `atomicMin` on `cachedLevel`. Rasterization only, so it carries none of the ray tracing
  branches, the same way [`traversal_init_blas_reuse.comp.glsl`](../shaders/traversal_init_blas_reuse.comp.glsl) carries
  none of the rasterization ones.
* [`traversal.glsl`](../shaders/traversal.glsl): `classifyInstanceLod` is shared with
  [`instance_classify_lod.comp.glsl`](../shaders/instance_classify_lod.comp.glsl), which does the equivalent job for
  BLAS sharing and caching. `unpackTraversalDiscrete` splits the tag off an enqueued instance ID.
* [`traversal_run.comp.glsl`](../shaders/traversal_run.comp.glsl) and
  [`traversal_run_groups.comp.glsl`](../shaders/traversal_run_groups.comp.glsl): a tagged node is always descended, a
  tagged group always draws its clusters.
* [`streaming.glsl`](../shaders/streaming.glsl): `streamingAgeFilter` keeps the relied upon levels alive, shared with
  BLAS caching.
* [`scene_streaming.cpp`](../src/scene_streaming.cpp): check the `SceneStreaming::handleDiscreteLod` function, and note
  that the geometry patch path is enabled for `useBlasCaching || useDiscreteLod`.
