# Scene Processing

How the sample turns an input glTF scene into the cluster LoD data it renders from, and how that result is cached.
For the LoD data structures themselves see [Continuous Cluster LoD Generation](lod_generation.md).

## Per-mesh simplification overrides

The simplification settings are global, which is not always enough: foliage wants different weights than a solid surface. `--simplifyoverrides <file.json>` overrides them per mesh. The file is an array of entries, each with a `"mesh"` regular expression matched against the glTF mesh name, plus the settings it replaces. The setting names are the same as the equivalent command-line options:

```json
[
  { "mesh": "leaves_.*",  "simplifyuvweight": 1.0, "simplifydilateall": true },
  { "mesh": "leaves_lod", "simplifyuvweight": 0.25 }
]
```

All matching entries are applied in file order, so a later entry wins over an earlier one. That lets you write a broad rule first and narrow exceptions after it, as above.

These settings can be overridden (see [scene_simplify_overrides.cpp](/src/scene_simplify_overrides.cpp)), the descriptions match their command-line equivalents:

* `simplifynormalweight` - weight of normals in the simplification error metric, 0 disables. default 0.5
* `simplifyuvweight` - weight of texcoords in the simplification error metric, 0 disables. default 0.5
* `simplifytangentweight` - weight of tangents in the simplification error metric, 0 disables. default 0
* `simplifytangentsignweight` - weight of the tangent sign in the simplification error metric, 0 disables. default 0.2
* `simplifymaterialweight` - weight of the material in the simplification error metric, 0 disables. default 0.1
* `loderrormergeprevious` - lod error propagation: scales the previous error in `max(previous * factor, error)`. >= 1, default 1.5
* `loderrormergeadditive` - lod error propagation: adds this much of the current error after the maximum. default 0
* `loderroredgelimit` - limit the lod error by edge length, to drop subpixel triangles despite high attribute error. default 1
* `simplifyerrorclamped` - clamp the attribute error to the position error scale, avoids overly conservative lod picking. default true
* `simplifypreservefolds` - try to keep fold lines between opposite-facing triangles, costs a bit of processing time. default false
* `simplifydilateall` - dilate open cluster borders to compensate the area loss of simplification, meant for foliage rather than everything. default false
* `simplifydilatetwosided` - enable `simplifydilateall` for geometries that use a two-sided material, which foliage typically does. default true
* `optimizeclusterslevel` - triangle order within a cluster, higher trades processing time for compression ratio, 0 to 3. default 1

Meshes that simplify differently no longer deduplicate into a single geometry, and the applied overrides are part of the cache file's per-geometry validity check, so editing the file re-processes the affected meshes.

> [!WARNING]
> The processing of larger scenes can take a while, even on CPUs with many cores. Therefore the application will save
> an uncompressed cache file of the results automatically. This file is a simple memory mappable binary file that can take a lot of space
> and is placed next to the original file with a `.nvsngeo` file ending. During processing existing cache files will be overwritten without warning.
>
> With the `--processingonly 1` command-line option one can reduce peak memory consumption during processing of scenes with many geometries.
> In this mode saving to the cache file is interleaved with the processing and resources are deallocated immediately once saved.
> At the end of the processing the app closes automatically.
> In combination with the `--processingpartial 1` command-line option, the processing only mode can resume partial results. So one can terminate the app during processing and continue at a later time.
> There is no consistency checking of settings or input meshes for this.
> To reduce system resource usage during processing use: `--processingthreadpct <float 0.0 - 1.0>` (default is 0.5, half the systems supported concurrency) and `--processingmemorygigabytes < ==0 is default of 60%, <0 is percentage (-60 means 60%), >0 is absolute >`
>
> If system memory usage after loading a cached file is a concern, then `--mappedcache 1` can be used to load data through memory mapping directly (forced for caches that are >= 2 GiB). However, we still have to improve the streaming logic a bit to avoid IO related hitches.
>
> Be aware, there are currently only few compatibility checks for these cache files, therefore we recommend deleting if changes were made to the original input mesh.
