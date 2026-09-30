# Command-line Options

`vk_lod_clusters --help` prints all registered options with their descriptions, this page only highlights the
noteworthy ones. Options can also be put into a `.cfg` file and passed with `--configfile <file.cfg>`.

The `scene_overrides` directory next to `shaders` holds per-scene settings that ship with the sample instead of
with the model. Loading `<name>.gltf`/`.glb`/`.cfg` also runs `scene_overrides/<name>.cfg` if it exists, after the
scene's own `.cfg` and before the scene is loaded, so options that affect loading take effect. An override cannot
change `--scene`.

* `--renderer 0` starts with rasterization.
* `--supersample 0` disables the super sampling that otherwise doubles rendering resolution in each dimension. 
* `--clasallocator 0` disables the more complex gpu-driven allocator when streaming and uses the simple move based CLAS compaction instead. It moves all resident CLAS whenever a group is unloaded, which makes `--blascaching` ineffective while it is used, see [BLAS Caching](blas_caching.md).
* `--gridcopies N` set the number of model copies in the scene.
* `--gridunique 0` disables the generation of unique geometries for every model copy. Greatly reduces memory consumption by truly instancing everything. It's on for the sample bunny scene by default, but off otherwise.
* `--streaming 0` disables streaming system and uses preloaded scene (warning this can use a lot of memory, use above `--gridunique 0` to reduce)
* `--vsync 0` disable vsync. If changing vsync via UI does not work, try to use the driver's *NVIDIA Control Panel* and set `Vulkan/OpenGL present method: native`.
* `--autoloadcache 0` disables loading scenes from cache file.
* `--mappedcache 1` keeps memory mapped cache file persistently, otherwise loads cache to system memory. Useful to save RAM on very large scenes.
* `--autosavecache 0` disables saving the cache file.
* `--meshoptarena 0` disables the per-thread stack arena that serves `meshoptimizer`'s temporary allocations during cluster building. The arena is on by default; it removes most of the global allocator contention that otherwise limits how well lod processing scales with thread count. `--meshoptarenabudget <MiB>` caps how much each thread keeps between calls (default **8**), below which the arena starts thrashing on chunk churn.
* `--forcepreprocessmegabytes 1024` if a scene's raw geometry (vertex & indices) is greater than this cutoff, use a dedicated preprocess pass. Can be quicker and allows using memory mapped cache file. Default is 2048 for 2 GiB.
* `--multimaterials 1 --attributes 7 --texturedmaterials 1` enables textured PBR materials (see [Materials](materials.md) below).
* `--maxtexturemegabytes <MiB>` sets an upper VRAM budget for material textures (default **4096**; **0** = no limit). See [Materials](materials.md).
* All megabyte budgets (`--maxgeomegabytes`, `--maxclasmegabytes`, `--startclasmegabytes`, `--clasgrowmegabytes`, `--maxblascachingmegabytes`, `--maxtransfermegabytes`, `--maxtexturemegabytes`) also accept a **negative** value, which is interpreted as a percentage of the device local heap. For example `--maxgeomegabytes -10 --maxclasmegabytes -10` reserves 10 % of VRAM each, independent of the GPU in use.
* `--dlss 1` enables DLSS when built with `USE_DLSS` (Super Resolution in rasterization, denoising in ray tracing). Use `--dlssquality <0-3>` to set quality (max performance through ultra performance).
* `--camerastring "..."` sets the initial camera (copy/paste from the _Misc Settings → Camera_ widget).
* `--addcamerapath "..."` defines a camera fly-through path (repeatable, see [Camera Path](camera_paths.md)).
* `--loadcamerapaths "file.txt"` replaces all paths with the definitions from a text file (see [Camera Path](camera_paths.md)).
* `--runcamerapath <index> <framecount>` deterministically plays back a defined path for benchmarking (see [Camera Path](camera_paths.md)).
