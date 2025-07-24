# Changelog for vk_lod_clusters
* 2025-7-24:
  * Major feature update.
  * Added ["BLAS sharing"](docs/blas_sharing.md) under "Traversal" to drastically reduce BLAS builds when geometries are instanced a lot. It is automatically enabled (`--autosharing 0/1`) for scene configurations that may benefit.
  * Added ""Culled error scale" under "Traversal" to allow more control over the lod error allowed in indirectly visible instances (`--cullederrorscale <float>`).
  * Added "Rendered Statistics" option to enable/disable computation of rendering statistics under "Traversal", this was always on before but had quite the performance impact and is now disabled.
  * Allow changing of cluster & lod settings for files that were loaded with a cache, triggers processing again.
  * Auto enable storing the file cache.
* 2025-7-18:
  * Mirror Box: Double right-click or M key to investigate out of frustum / occluded object behavior. Under the "Settings" tab this box can be adjusted manually as well.
  * Improve ray offsets for shadows/ao in large scenes to avoid self-intersection.
* 2025-7-17:
  * Blas Sharing. Significantly reduces BLAS builds in ray tracing by sharing BLAS among instances. See new documentation TODO.
* 2025-7-16:
  * Bugfix crash for ray tracing when not all tlas instances had a valid blas. This could have happened when we ran out of renderable clusters. The fix ensures the use of the low detail blas.
* 2025-7-14:
  * Improve streaming request and unload logic. The logic was moved after ensuring a positive node traversal decision. Otherwise too many false positives were generated.
* 2025-7-11:
  * The downloadable extra scenes `threedscans_animals` and `threedscans_statues` were updated. By accident they were stored with independent triangles which increased storage and processing time unnecessarily.
  * Change default cluster config to 128 triangles 128 vertices (more common).
  * Add `Geometries` to "Statistics" ui.
  * Updated `nv_cluster_lod_library` submodule, which brings performance improvements to "inner" parallelism, the `threedscans_animals` processing improved from 26 to 11 seconds on a windows test system.
* 2025-7-10:
  * Reduce number of BLAS builds by introducing a per-geometry `lowDetailBlas` that is built once at scene preparation time.
    A new compute shader step was added [shaders/tlas_instances_blas.comp.glsl](/shaders/tlas_instances_blas.comp.glsl) and
    the number of BLAS build operations is now sourced indirectly from the GPU.
  * Reduce system memory consumption greatly after building the cluster and lod data.
* 2025-7-9:
  * Fix regression of `nvpro_core2` port: culling parameter was not properly hooked up
* 2025-7-7:
  * Fix regression of `nvpro_core2` port: sky light intensity & missing sky UI
  * Note: `nvpro_core2` must be a version that already includes fixes to `nvutils::IDPool` and `nvvk::BufferSubAllocator` to fix some instabilities during streaming.
  * Add basic accessor-based detection of unique geometries within gltf meshes.
  * Add simple colored material import from gltf.
* 2025-7-4:
  * Add missing file menu entries (lost during framework porting)
  * Add file menu to delete the cache file.
* 2025-7-3:
  * Change to `--mappedcache 0` default. Loads cachefiles into system memory for faster access during streaming.
  * Bugfix loading cache file introduced by last update
* 2025-6-27
  * Updated `nv_cluster_lod_library` submodule, which has new API & proper vertex limit support.
* 2025-6-26:
  * Ported to use `nvpro_core2`, as result command-line arguments are now prefixed with `--` rather than just `-`. It is recommended to delete existing /_build or the CMake cache prior building or generating new solutions.
* 2025-6-2:
  * Scale traversal lod error internally based on super-sampling. Significantly speeds things up compared to old defaults which caused much more geometry to be loaded and rendered.
  * Skip traversal and directly enqueue lowest detail cluster in `traversal_init.comp.glsl` when only lowest lod is used.
* 2025-5-30:
  * Added early detection if only last lod level is required in `traversal_init.comp.glsl`
  * Changed persistent kernel thread count heuristic. Works much better on larger scenes.
* 2025-4-26:
  * Added "Disable back-face culling" to "Scene Complexity" UI.
* 2025-4-25:
  * Added "Instance Sorting" option, does sort instances by distance to camera. `--instancesorting 0/1`.
  * bugfix gltf meshes with multiple primitives
* 2025-4-23:
  * Add `--processingthreadpct <float 0-1.0>` to control the percentage of threads doing the geometry processing (number of geometries in parallel). Percentage of what the system supports for concurrency. Default is `0.5`.
  * Add `--processingonly 0/1` to reduce peak memory consumption during processing and saving the cache file. This always saves a cache file (unless the old one was valid) and terminates the application afterwards.
* 2025-4-11:
  * Interleave geometry processing with loading to reduce peak memory consumption.
  * Add visualization of instance bounding boxes
* 2025-4-7:
  * Bugfix to file cache header detection.
  * The file cache can be used via memory mapping, avoiding a copy into system memory. `--mappedcache 0/1` defaults to true.
  * Use "octant" encoding for vertex normals according to [A Survey of Efficient Representations for Independent Unit Vectors](http://jcgt.org/published/0003/02/01/paper.pdf)
* 2025-4-4: 
  * The file cache format now stores everything geometry related for rendering. Instance and material information, as well as original vertex/triangle counts still comes from the gltf. The new file ending is `.nvsngeo`, the old `.nvcllod` files no longer work.
  * Added `--autoloadcache 0/1` option to disable loading from a cache file.
  * Some basic preparation to allow working from memory mapped cache files without loading into system memory.
* 2025-2-7:
  * Added _"File > Save Cache"_ menu entry, as well as `--autosavecache 1` option. This allows to store the results of the lod cluster mesh processing into a file next to the original model.
    This allows speeding up future load times of the model a lot. See new notes in **Model processing** section of README
  * Improved warnings and some memory statistics.
  * Streaming geometry memory now guaranteed to stay within limit.
* 2025-1-30: Initial release