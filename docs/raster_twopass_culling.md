# Two-Pass Culling for rasterization with cluster-based continuous level of detail

The occlusion culling within the LoD traversal tests the bounding spheres and boxes of instances, nodes and clusters against a HiZ
(hierarchical depth) buffer.

The sample provides three variants for how to do this HiZ testing, all of them are rasterization only and configured in the UI under
_"Traversal"_.

## Motivation

The cheap answer is to test against the HiZ of the last frame, using last frame's matrices. It costs nothing extra, but
geometry that becomes visible through camera motion or disocclusion is missing for a frame, which is noticeable on
faster motion.

Two-pass culling fixes that by rendering what was visible before, rebuilding the HiZ from it and then testing again. The
assumption behind it is: what was visible last frame is very likely visible now, so those surfaces are good
occluders for everything else. The downside is that the second pass repeats the entire pipeline. By using reject
lists we can speed this second pass up.

> [!NOTE]
> This has been in use for a while.
> [Nießner and Loop 2012, "Patch-based Occlusion Culling for Hardware Tessellation"](http://www.niessnerlab.org/projects/niessner2012patch.html)
> applied it to hardware tessellation and already kept lists of visible and occluded patches across frames.
> [GPU-Driven Rendering Pipelines (Haar and Aaltonen, SIGGRAPH 2015)](https://www.advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf)
> brought it into a GPU-driven cluster pipeline as "two-phase occlusion culling", and
> [Nanite (Karis et al., SIGGRAPH 2021)](https://www.advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf)
> uses it as well.

## Variants

### Single pass

The default. Traversal tests against the HiZ of the last frame with last frame's matrices, and then renders.

### Two-pass culling

_"Use TwoPass Culling"_ (`--twopassculling 1`) renders in two passes per frame:

1. The first pass traverses and renders against last frame's HiZ. This yields everything that was visible before.
2. We rebuild the HiZ from the depth buffer that the first pass produced, it now represents the current frame.
3. The second pass traverses again, this time against the new HiZ, and renders only what was not already drawn in the first pass.

We keep two HiZ layers alive for this (`Resources::m_hizUpdate`). `texHizFar[0]` is last frame's final HiZ,
`texHizFar[1]` the one that was built after the first pass.

This has two disadvantages. The second pass repeats the work of the first one: `traversal_init` runs over every instance
again, the LoD hierarchy is traversed from the root nodes again and as a result things are tested twice, once against the old
HiZ to detect that it was rendered in the first pass already and once against the new HiZ. And because that detection
needs both HiZ at the same time, we have to keep two layers around.

### Two-pass culling with reject lists

_"TwoPass Reject Lists"_ (`--twopassrejectlists 1`, enabled by default) avoids that repeated work, following what Nanite
does: its culling dataflow passes "occluded instances" and "occluded nodes and clusters" from the main pass to the post
pass.

The traversal matrix and the error threshold are the same in both passes, so all LoD decisions are identical and only
the visibility result can differ. The first pass therefore records what the LoD metric wanted but the visibility test
rejected, and the second pass only repeats the visibility test:

* instances, as a compact list that the second pass runs `traversal_init` over indirectly
* inner nodes, which seed the node queue instead of the root nodes
* cluster groups, which seed the group queue
* per traversed cluster group a bitmask of its rejected clusters

A cluster rendered in the first pass has no bit set, so the second pass no longer needs the test against the old HiZ to
detect that it was drawn already. A single HiZ layer would therefore be enough here. We keep two for simplicity, they
are always allocated in `Resources::updateFramebufferRenderSizeDependent`.

`setupSecondPass` points `SceneBuilding::traversalNodeInfos` and `SceneBuilding::traversalGroupInfos` at the reject
arrays and seeds their counters, the traversal kernels themselves stay unchanged.

> [!IMPORTANT] Everything is filtered against the current frustum before it is recorded, with `intersectFrustumOnly`.
> The new HiZ does not exist yet during the first pass, but the current matrices do, and what is outside that frustum
> cannot reappear. Without this we would record nearly every instance that is not visible and gain nothing. For the
> cluster bitmasks we test the object-space union of a group's rejected clusters once.

## Results and limitations

Almost all of the benefit comes from the second pass no longer running `traversal_init` over every instance, so the
technique scales with the number of instances and not with the depth of the LoD hierarchies.

The cost is another node and group queue of `1 << rendertraversalbits` entries plus the cluster bitmasks, and some extra
work in the first pass for the recording and the frustum test. Overflow of the reject lists is clamped and reported like
the other traversal limits.

With two-pass culling active the profiler reports the two passes separately, as _"Traversal Preparation 0/1"_,
_"Traversal Run 0/1"_ and _"Draw 0/1"_. A benchmark sequence additionally reports _"Reject Nodes"_, _"Reject Groups"_ and
_"Reject Instances"_, which show how much work the first pass handed over.

## Implementation

In the source look for `USE_TWO_PASS_CULLING` and `USE_TWO_PASS_REJECT_LISTS`. Both are pre-anded with the culling
option by the host, so they can be tested on their own.

Key changes in the shaders and device code are:
* [`culling.glsl`](../shaders/culling.glsl): `intersectFrustumOnly` is the frustum test used for the recording, it skips
  the clip-space bounds that are only needed for the HiZ lookup itself.
* [`traversal.glsl`](../shaders/traversal.glsl): `queryClusterWasVisible` drops the test against the old HiZ, a single
  test against the pass' own HiZ is enough once the reject bits carry the first pass' answer.
* [`traversal_init.comp.glsl`](../shaders/traversal_init.comp.glsl): records the rejected instances and, in the second
  pass, iterates the compacted list instead of all instances.
* [`traversal_run.comp.glsl`](../shaders/traversal_run.comp.glsl): records rejected inner nodes and group leaves into
  the two seed arrays.
* [`traversal_run_groups.comp.glsl`](../shaders/traversal_run_groups.comp.glsl): builds the per-cluster reject bitmask
  of a group, and does the residency check for the group leaves that the second pass seeds directly.
* [`traversal_reject_clusters.comp.glsl`](../shaders/traversal_reject_clusters.comp.glsl): the second pass kernel that
  re-tests the recorded clusters, it does no traversal and no metric evaluation.
* [`build_setup.comp.glsl`](../shaders/build_setup.comp.glsl): `setupSecondPass` swaps the queues over to the reject
  arrays and sets up the indirect dispatches for the second pass.
