# Changelog for vk_lod_clusters
* 2026-4-26:
  * Added "Disable back-face culling" to "Scene Complexity" UI.
* 2026-4-25:
  * Added "Instance Sorting" option, does sort instances by distance to camera. `-instancesorting 0/1`.
  * bugfix gltf meshes with multiple primitives
* 2026-4-23:
  * Add `-processingthreadpct <float 0-1.0>` to control the percentage of threads doing the geometry processing (number of geometries in parallel). Percentage of what the system supports for concurrency. Default is `0.5`.
  * Add `-processingonly 0/1` to reduce peak memory consumption during processing and saving the cache file. This always saves a cache file (unless the old one was valid) and terminates the application afterwards.
* 2026-4-11:
  * Interleave geometry processing with loading to reduce peak memory consumption.
  * Add visualization of instance bounding boxes
* 2026-4-7:
  * Bugfix to file cache header detection.
  * The file cache can be used via memory mapping, avoiding a copy into system memory. `-mappedcache 0/1` defaults to true.
  * Use "octant" encoding for vertex normals according to [A Survey of Efficient Representations for Independent Unit Vectors](http://jcgt.org/published/0003/02/01/paper.pdf)
* 2025-4-4: 
  * The file cache format now stores everything geometry related for rendering. Instance and material information, as well as original vertex/triangle counts still comes from the gltf. The new file ending is `.nvsngeo`, the old `.nvcllod` files no longer work.
  * Added `-autoloadcache 0/1` option to disable loading from a cache file.
  * Some basic preparation to allow working from memory mapped cache files without loading into system memory.
* 2025-2-7:
  * Added _"File > Save Cache"_ menu entry, as well as `-autosavecache 1` option. This allows to store the results of the lod cluster mesh processing into a file next to the original model.
    This allows speeding up future load times of the model a lot. See new notes in **Model processing** section of README
  * Improved warnings and some memory statistics.
  * Streaming geometry memory now guaranteed to stay within limit.
* 2025-1-30: Initial release