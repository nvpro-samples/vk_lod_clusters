# Render permutation tests

**155 sequences in 8 files** for a full pass with the validation layers on. Most of the
time is pipeline compilation, not rendering.

Parameter sequences that drive the sample through the permutations that are easy to break
and easy to miss. There is **no golden image comparison** - the question is whether every
permutation still builds its pipelines and renders, not whether it renders the same pixels
as last week. Screenshots are written so a failure can be looked at, not compared.

Bugs these have found are fixed as they come up, the CHANGELOG records them.

## Scope

These cover the **rendering** side: renderers, traversal, culling, streaming, BLAS handling,
visualization modes. They all run against an already processed scene, mostly the bundled
bunny, and a cached one at that, so they say nothing about cluster generation, simplification
or the lod build. Changing anything in mesh processing is not what this suite verifies - run
the affected scene through the processing directly instead (`--autoloadcache 0` so the cache
does not hide the change), and add permutations here only when the change reaches the renderer.

## Running

```bash
tests/run_tests.sh
```

```powershell
tests\run_tests.ps1
```

Both run every `seq_*.txt` here as its own headless invocation, writing `output.log`, the
`sequence.txt` that actually ran, and `screenshot_<index>_<sequence name>.jpg` into
`tests/_results/<timestamp>/<sequence file>/` (gitignored).

| | PowerShell | bash |
| --- | --- | --- |
| pick the binary | `-Config Debug`, `-Exe <path>` | `-e <path>` |
| frames per sequence (default 32) | `-SequenceFrames 16` | `-f 16` |
| run one file | `-Sequences seq_visualize.txt` | trailing `seq_visualize.txt` |
| skip screenshots | `-Screenshot 0` | `-c 0` |
| validation layers (default **on**) | `-Validation 0` | `-V 0` |
| validation preset (default standard) | `-ValidationPreset 4` | `-P 4` |
| override the scene for every file | `-Scene <cfg>` | `-s <cfg>` |

Validation is on by default because these are correctness runs - with the layers off the
`VUID-` scan below can never find anything.

**A sequence file fails** when the process exits non-zero, when it produced fewer
screenshots than it has sequences, or when the log contains `shaders failed`, `error:`,
`ERROR:`, `VUID-`, `Validation Error`, `DEVICE_LOST` or `failed to allocate`. The exit code
alone proves nothing: a pipeline that fails to build is logged and then presented as a black
frame.

## Files

| file | sequences | covers |
| --- | --- | --- |
| `seq_visualize.txt` | 35 | all 12 visualization modes on both renderers, compute rasterization, the DLSS shader permutations, path tracing |
| `seq_raster_features.txt` | 29 | `--culling` x `--twopassculling` x `--discretelod` over both geometry layouts, plus primitive culling, EXT mesh shader, compute rasterization, streaming off, HBAO, bbox overlays, render stats |
| `seq_materials.txt` + `.cfg` | 23 | vertex attribute sets, textured materials, alpha mask and blend, two sided, multi material, texture lod modes, forced compute rasterization, `--compressed`, path tracing on/off |
| `seq_raytrace_blas.txt` | 19 | `--blassharing` x `--blasmerging` x `--blascaching` over both geometry layouts, plus render stats |
| `seq_limits.txt` + `.cfg` | 17 | every limit the sample warns about rather than enforces, exceeded one at a time and together |
| `seq_traversal.txt` | 12 | `--persistenttraversal` off/on against both renderers, crossed with instance sorting, discrete lod and culling |
| `seq_streaming.txt` | 10 | streaming against preloaded, the CLAS allocator, async and decoupled transfers |
| `seq_16bit_dispatch.txt` + `.cfg` | 10 | the 16 bit compute launch grid, on its own denser scene |

A sequence file can bring its own scene as `<same name>.cfg`; the scripts pick that up
automatically. That is for settings a sequence cannot change on its own because they are
only read at startup.

## Scenes

**`bunny_grid.cfg`** (the default) - 256 bunnies on a grid, framed whole by the fit-to-scene
camera, so one view spans lod 0 up to the coarsest levels. The defaults are not usable as a
fixture: without an explicit `--scene` the sample forces unique geometries onto the built-in
bunny and sizes the grid from the device heap, and `--autosharing 0` is essential or the
sharing heuristic overwrites half the BLAS matrix.

`--gridunique` is deliberately not set, the sequences toggle it: with `0` all 256 instances
share one geometry, which is what BLAS sharing is about; with `1` every copy owns its
geometry, which is what stresses streaming, merging and caching.

**`seq_materials.cfg`** - `materialtest.gltf`, a row of subdivided boxes carrying the mesh
and material states the sample branches on. The bunny is POSITION + NORMAL and one opaque
material, so without this fixture `ALLOW_VERTEX_TANGENTS`, `ALLOW_VERTEX_TEXCOORD_0/_1`,
`HAS_TEXTURED_MATERIALS`, `HAS_ALPHA_TEST` and `USE_TWO_SIDED` never compile at all.
Regenerate it with `python tests/make_materialtest.py`; it processes in about a millisecond,
so it runs with `--autoloadcache 0` rather than risking a stale cache.

**`seq_16bit_dispatch.cfg`** - `--force16bitdispatch` takes the `USE_16BIT_DISPATCH` path on
hardware that does not need it. That covers the compile; reaching the grid conversion, the
padded trailing work groups and the linearized work group index needs counts above 65536,
hence 16384 instances and `--loderror 0.1`.

`--swrasterthreshold 100000` in that file forces every cluster through the compute
rasterizer instead of the handful that are small enough by default, which is the only way
its alpha test branch is reached.

**`seq_limits.cfg`** - 1024 unique geometries against budgets that cannot hold them.
Exceeding a limit is a normal runtime state, so the expected outcome is a degraded image and
nothing else. The screenshots are meant to be visibly incomplete.

## Adding a permutation

Sequence files are [parameter sequences](../docs/camera_paths.md): `SEQUENCE "name"` then
the parameters it changes. The name becomes the screenshot filename, so keep it to
`[a-z0-9_]`.

Parameters **persist across sequences** - set everything a sequence depends on explicitly,
or reordering changes the result. `--sequenceframes` only exists once the sequencer has
initialized, so it cannot be passed on the command line; it sits in the first `SEQUENCE` of
each file and the scripts rewrite it when `-SequenceFrames` / `-f` is given.

Some combinations are resolved rather than rejected: `--swraster` needs `--culling 1` and
visualization `2` or `10`, `--blasmerging` / `--blascaching` need `--streaming 1`,
visualization `2` / `10` force shading off. Keep them in anyway - an inapplicable
combination has to be a harmless no-op, not a failure.
