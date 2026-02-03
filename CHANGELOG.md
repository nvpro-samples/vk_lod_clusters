# Changelog for vk_lod_clusters
* 2026-2-3:
  * Bugfix freeze culling with "Two Pass Culling" and manage the matrix assignment outside the renderers.
  * Bugfix regression when applying cluster changes in ui due to new cachesuffix changes.
* 2026-1-30:
  * Performance fix in `Scene::decompressGroup` accidental reads on write-combined memory (thanks Arseny Kapoulkine). This impacted streaming performance when using compressed data quite a lot. In future we intend to decompress on GPU.
  * Bugfix regression with `Allow SW-Raster`, forgot changes for "reverse Z".
* 2026-1-29:
  * Added `--headless` and `--headlessframes <int frame count>` options to run benchmarks without window creation.
  * Added `--sequencescreenshot <mode: 0 default off, 1 window, 2 viewport>` to store a screenshot with each completed scripted sequence.
  * Added `--cachesuffix <string default ".nvsngeo">`
* 2026-1-23:
  * Added scripted benchmark mode: `--sequencefile <filename>` leveraging `nvutils::ParameterSequencer`. Pass a text file with command-line arguments. Each sequence starts with a line like `SEQUENCE "sequence name"` followed by command-line arguments that define the settings used.
* 2026-1-12:
  * Added `Force Invisible Culling` option to "Traversal -> Other" for ray tracing when culling is enabled. This will remove instances and clusters that aren't primary visible.
  It allows to get the same scene complexity as rasterization in both streaming and rendering complexity.  
  Note that it does create artifacts in secondary rays due to missing geometry, and BLAS sharing/merging may also cause missing geometry (technically only BLAS caching is valid). Furthermore, it may yield missing geometry under motion, as it lacks a "Two Pass Culling" equivalent.
* 2026-1-8:
  * Updated SSAO implementation, taken from [NVIDIA-RTX/Donut](https://github.com/NVIDIA-RTX/Donut)
  * Use "reverse Z" to improve depth precision in rasterization.
* 2025-12-15:
  WARNING: Needs updated nvpro_core2
  * Added `Two Pass Culling` to Traversal options, to do proper temporal coherent occlusion culling for rasterization.
    This fixes the artifacts on the screen borders on rapid movement.
    The first pass uses the previous frame's HiZ to do the occlusion culling. After its rendering a new HiZ is built.
    The second pass renders everything that wasn't rendered in the first pass (passes old HiZ) and passes the new HiZ.
    After the second pass the HiZ is updated again, to be used for the next frame.
  * Support two-sided materials in `EXT_mesh_shader`, note this is a lot slower than `NV_mesh_shader` due to use of shared memory to dynamically flip the winding to be always front-facing. The alternative is to use two PSOs, with different rasterization state and then bin clusters accordingly.
  * Added `depth only (black)` visualize mode to emulate shadow map rendering.
  * Split `Culling / LoD Freeze` into `Freeze Lod` and `Freeze Culling`, also for shadow map style rendering emulation.
  * Added a few fixed size square texture modes for super resolution.
* 2025-12-11:
  * Bugfix in ray tracing's flip winding handling (regression from previous change). No need to set the flag, as ray tracing handles negative determinants automatically.
  * Added support for glTF "doubleSided" material property handling.
  * Removed `twosided` command-line parameter (silently ignored as used by zorah config file), added `forcetwosided` instead.
  * Improved `primitive culling` using determinant as mentioned in https://zeux.io/2023/04/28/triangle-backface-culling/
  * Switched to `mat4x3` to keep `RenderInstance` in 128 bytes.
* 2025-12-10:
  * Updated `meshopt_clusterlod.h` to reflect new options from meshoptimizer v1
  * Added `primitive culling` support for `NV_mesh_shader`
  * Improved back-face culling, using flip of the triangle winding based on instance matrix determinant. (Material-based two-sided yet to be done).
* 2025-12-9:
  * Removed meshoptimizer as submodule, it is now part of nvpro_core2
  * Removed LoD pixel scale adjustment when using the fixed super resolutions (`720p`...)
  * Disable HBAO in `visibility buffer` mode
* 2025-12-2:
  * Added `Allow SW-Raster` option for compute-based rasterization and a few conditions (visibility buffer mode, culling on, separate groups on). Very crude basic implementation so far.
  * Added `720p`, `1080p` and `1440p` as "super sampling" modes to set fixed resolutions.
* 2025-12-1:
  * Added `visibility buffer` visualization mode that turns off all shading.
* 2025-11-25:
  * Refactored glTF loading to use dedicated `Scene::loadGeometryGLTF`
  * Added support for glTF `EXT_meshopt_compression`
* 2025-11-13:
  * Skip file mapping glTF buffers when file cache is used
  * Bugfix shader compilation bug
  * Add note about tangent space encoding
* 2025-11-10:

  **WARNING** Old cache files are not be compatible anymore. First time loading such scenes will trigger processing and overwrite / delete them.

  * Removed `nv_cluster_lod_library` usage and dependency, `meshoptimizer`'s cluster lod builder is now the only implementation and enabled the removal of some abstractions.
  * Added [documentation](docs/lod_generation.md) about cluster lod generation, that originated from the nv library.
  * Combined vertex normal and tangent to single 32-bit value.
  * Revised UI around cluster settings
  * Revised progress bar for processing/loading to be based on triangle count, not geometry count
  * Added second texture coordinate attribute (unused so far)
  * Added histograms to scene cache file for faster loading times (avoids iterating clusters).
  * Added support for basic cluster group compression. See `Scene::compressGroup` within [scene_cluster_compression.cpp](src/scene_cluster_compression.cpp). Position and texcoord floats are encoded lossless, however, we recommend combining it with dropping mantissa bits for better compression. New command-line options are `--compressed 0/1`, `--compressedpositionbits <dropped bits>` and  `--compressedtexcoordbits <dropped bits>`. Currently decompression is done prior upload via `Scene::decompressGroup`, however we intend to do this in a compute shader after upload, to reduce the streaming traffic.

* 2025-10-30:

  **WARNING** Old cache files are not be compatible anymore. First time loading such scenes will trigger processing and overwrite / delete them.

  * Major refactoring of the scene cache file. It now stores the runtime data so that it can be easily streamed in
    using a single binary blob for the whole group. The `shaderio::Cluster` and `shaderio::Group` data structures
    were changed to allow more optional vertex attributes, as well as ease compatibility with HLSL / byte offsets
    and not relying on 64-bit VAs.

    Further work is being done on compressed representations for the disk cache that are decoded by compute shaders, and will be the next bigger update.
  * Spatial sorting of cluster groups within a lod level to help with streaming locality. Thanks to
    Arseny Kapoulkine for the `partition_sort` option that was added to `meshopt_clusterlod.h`
  * Preparations for enhanced materials have been done. UV and tangent space vertex attributes were added.
    All attributes are taken into account during mesh simplification according to meshoptimizers weight handling. The weights can be set by command-line, for example `--simplifynormalweight 1.0`. To ignore all attribute loading use `--attributes 0`. Default value is `1` which means only vertex normals are enabled (see `shaderio::ClusterAttributeBits`). Later versions will add more material features, such as texture loading etc.
  * Moved optional vertex attribute loading into fragment shader. `VK_KHR_fragment_shader_barycentric` is now required for proper shading. This keeps the mesh shader smaller in its output size.
* 2025-10-17:
  * Added `Cluster BBoxes` visualization to "Rendering -> Other settings". Note the bounding box visualizations don't work for ray tracing when DLSS is active, and they will only show clusters that are part of BLAS builds in that frame.
  * Bugfix supersampling change in UI causing some rendering artifacts.
  * Reduce risk of out of memory in the local modified meshoptimizer clusterlod builder. Rarely hit a case at the end of the mip chain where more clusters were generated after simplification than input cluster count.
  * Added option to compute simple metric for occupancy within cluster bounding boxes.
  * Expose `RA split factor` in "Clusterization -> Other settings" to influence the correspondign value in meshoptimizer's clusterizer when it is set to prefer rasterization.
* 2025-10-8:
  * Updated meshoptimizer to use its [improved partitioner](https://github.com/zeux/meshoptimizer/pull/964) that supports spatial partitioning. As a result removed the `partition_spatial_average` logic from our local version.
* 2025-9-30:
  * Improved processing of large scenes through the ordering of the processing of geometries by descending triangle counts. This yields better work distribution across threads. Thanks to Arseny Kapoulkine for this suggestion. As result the zorah scene can be processed in around 6 minutes on a 16-core AMD Ryzen 9.
  * added `partition_spatial_average` to the local [modified meshoptimizer clusterlod builder](src/meshopt_clusterlod.h). Improves number of clusters per group, which is better for streaming in this sample and also increases the chance that the last lod level is a single cluster, which is mandatory for this sample.
* 2025-9-25:
  * All new cluster lod hierarchy builder based on [meshoptimizer's clusterlod.h](https://github.com/zeux/meshoptimizer/blob/master/demo/clusterlod.h). It is about 5x faster and 20x less memory during processing, also deterministic and the new default. Special thanks to Arseny Kapoulkine. This sample still uses a slightly [modified version](src/meshopt_clusterlod.h) of his work, to allow optional parallel processing. At the moment of writing `nv_cluster_lod_library` does perform better on meshes made of topology with little connectivity (leaves, rubble).
  * Moved cluster lod building into [scene_cluster_lod.cpp](src/scene_cluster_lod.cpp).
  * New version of `nv_cluster_lod_library` that no longer has `meshoptimizer` as git sub module, as result the `meshoptimizer` library is now a git sub module of this sample and was updated to a newer commit as well.
  * Larger scenes are preprocessed automatically with a dedicated pass during loading. Use `--forcepreprocessmegabytes 2048` to control this behavior. If a scene's raw geometry (vertex & indices) is greater than this cutoff, loading uses a dedicated preprocess pass. Can be quicker and allows using memory mapped cache file. Default is 2048 for 2 GiB.
* 2025-9-17:
  * Fix `VK_EXT_mesh_shader` respecting 16 bit grid dispatch limit.
  * Added support for 16 bit limit of gridX dispatch dimension for all compute shaders as well.
* 2025-9-16:
  * Added support for subgroup size 64 (untested).
* 2025-9-15:
  * Added `VK_EXT_mesh_shader` support (still default to NV if available).
  * Removed unnecessary `VK_NV_SHADER_SUBGROUP_PARTITIONED` requirements
  * Split README a bit.
* 2025-9-10:
  * Added ["BLAS Caching"](docs/blas_caching.md). This option enhances "BLAS Sharing" so that fully resident lod levels
    are kept in a dedicated BLAS that can be re-used many frames. Any instance whose minimum lod level is higher or 
    equal than this BLAS's lod level can use it. Therefore we can further reduce the number of BLAS built per frame.
  * Added ["BLAS Merging"](docs/blas_merging.md). This option enhances "BLAS Sharing" such that we build a single BLAS for 
    all instances that overlap in near lod range. The instances still make streaming requests, however, only a 
    single BLAS is built based on the highest available detail that is already streamed in. This technique therefore
    guarantees that at most only two BLAS are built per unique geometry in the scene. Which significantly helps
    reduce worst-case memory reservations and further reduces BLAS builds. Special thanks to Pyarelal Knowles for
    the idea to drive lod picking based on streaming state.
  * Enforce facet shading when scene has no vertex normals (compile shaders `ALLOW_VERTEX_NORMALS` accordingly)
  * Bugfix "CLAS position drop bits" option being ignored.
  * Bugfix detection when pre-loading is likely to overshoot device memory
  * Bugfix HiZ render target size with DLSS denoising active, which resulted in objects classifed as culled when they weren't.
  * Lot's of UI changes and added visible progress bar during scene loading.
  * Added `--processingpartial 1` command line option which allows the `--processingonly 1` mode to resume from partial results.
  * `--processingonly 1` mode now skips Vulkan context and application window creation.
  * `--processingmode <int>` 0 auto, -1 inner (within geometry), +1 outer (over geometries) parallelism. default 0.
  * Added `Scene::buildGeometryDedupVertices` which is triggered when we detect that a mesh as fully independent triangles (some exports do this).
  * Bugfix CMAKE USE_DLSS off case
* 2025-8-24:
  * Updated `meshoptimizer` submodule to `v 0.25`
* 2025-8-5:
  * DLSS denoiser support in ray tracing. Activate `USE_DLSS` in `cmake` to download and enable support within the application. 
* 2025-7-31:
  * Bugfix for objects with single lod level appearing black in lod visualization.
  * Filter out instances whose material uses BLENDED transparency.
  * Double-Click/SPACE also changes walk speed (percentage of distance to hit point).
* 2025-7-30:
  * Added "Separate Groups Kernel" optimization to "Traversal" (default true). See `USE_SEPARATE_GROUPS` in [shaders/traversal_run.comp.glsl](shaders/traversal_run.comp.glsl) as well as the new kernel [shaders/traversal_run_separate_groups.comp.glsl](shaders/traversal_run_separate_groups.comp.glsl).
  * Tweaked heuristic for persistent kernel threads once more.
* 2025-7-24:
  * Major feature update.
  * Added ["BLAS Sharing"](docs/blas_sharing.md) under "Traversal" to drastically reduce BLAS builds when geometries are instanced a lot. It is automatically enabled (`--autosharing 0/1`) for scene configurations that may benefit.
  * Added ""Culled error scale" under "Traversal" to allow more control over the lod error allowed in indirectly visible instances (`--cullederrorscale <float>`).
  * Added "Rendered Statistics" option to enable/disable computation of rendering statistics under "Traversal", this was always on before but had quite the performance impact and is now disabled.
  * Allow changing of cluster & lod settings for files that were loaded with a cache, triggers processing again.
  * Auto enable storing the cache file.
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
  * Bugfix to cache file header detection.
  * The cache file can be used via memory mapping, avoiding a copy into system memory. `--mappedcache 0/1` defaults to true.
  * Use "octant" encoding for vertex normals according to [A Survey of Efficient Representations for Independent Unit Vectors](http://jcgt.org/published/0003/02/01/paper.pdf)
* 2025-4-4: 
  * The cache file format now stores everything geometry related for rendering. Instance and material information, as well as original vertex/triangle counts still comes from the gltf. The new file ending is `.nvsngeo`, the old `.nvcllod` files no longer work.
  * Added `--autoloadcache 0/1` option to disable loading from a cache file.
  * Some basic preparation to allow working from memory mapped cache files without loading into system memory.
* 2025-2-7:
  * Added _"File > Save Cache"_ menu entry, as well as `--autosavecache 1` option. This allows to store the results of the lod cluster mesh processing into a file next to the original model.
    This allows speeding up future load times of the model a lot. See new notes in **Model processing** section of README
  * Improved warnings and some memory statistics.
  * Streaming geometry memory now guaranteed to stay within limit.
* 2025-1-30: Initial release