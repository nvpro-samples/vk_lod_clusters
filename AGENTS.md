# Notes for coding agents

How changes are integrated into this sample. The [README](README.md) and [docs/](docs) describe what the sample
does, this file how to work on it.

## Layout

* `src/`: host code. `lodclusters*.cpp` is the application (options, UI, sequences), `renderer_*.cpp` the
  renderers, `scene*.cpp` loading, processing, the cache file and streaming.
* `shaders/`: GLSL, compiled **at runtime** from this directory. `shaderio*.h` are shared between C++ and GLSL.
* `docs/`: one document per technique, linked from the README.
* `tests/`: parameter sequences that run the renderer permutations, see [tests/README.md](tests/README.md).

## Build and run

* CMake, a `Release` build is the one to test with. `nvpro_core2` is found next to the project or up to two levels
  above, otherwise downloaded into `build/_deps/nvpro_core2`.
* That download is not updated with the sample. When a build fails in or against `nvpro_core2` (missing functions,
  changed signatures), check that copy first: update it, or remove it so CMake downloads a current one.
* Sources are globbed with `CONFIGURE_DEPENDS`. With the Visual Studio generator the first build after files were
  added or removed regenerates the project but still fails, the second one works.
* Shaders compile when the renderer is created, from the source tree. The executable and the shaders must come from
  the same revision: updating the source tree under a running app, or running an old executable against new
  shaders, can hang the GPU. Rebuild after every update of the sources.
* Scripted runs: `--headless --headlessframes N --sequencefile file.txt --sequencescreenshot 2`. A sequence file is
  a list of `SEQUENCE "name"` blocks with the options that change, `--sequenceframes` and `--sequenceaverages`
  control how long each runs and what its timers average.

## Code

* Format `src/*.cpp`, `src/*.hpp`, `src/*.h` and `shaders/*.h` with clang-format 19 and the repository's
  `.clang-format`. CI checks it. The `.glsl` files are not formatted.
* Spell out types in new code, no `auto`. Leave existing code as it is.
* Comments are short and state what the code cannot: the reason, the invariant, the trap. Match the density of the
  file. No change history in comments.
* Features are switched by shader macros (`USE_...`) that the renderer sets from its config when it compiles the
  shaders. Renderer config changes recreate the renderer, frame config changes apply per frame.
* Structs in `shaderio*.h` use scalar layout. Change both sides together, keep fields that are read together in the
  same 128 byte line and debug or UI only fields apart from the hot ones.
* The streaming shaders are compiled once for all renderers. They must not depend on renderer macros, nor read
  buffers only a ray tracing renderer allocates.
* With `USE_16BIT_DISPATCH` large launch grids are rounded up to two dimensions. Work indexed by workgroup must
  guard against the overshoot. A 1D launch is sized exactly, so such guards can be compiled out there.
* A counter written by an earlier dispatch is read through the uniform buffer (`build`), the storage alias
  (`buildRW`) is only for atomics within a dispatch. Barriers name the uniform read when a uniform reads the result.
* Partition early-out: a skipped instance keeps the BLAS it had. Anything that can invalidate that BLAS or the
  clusters it was built from has to force all partitions to be processed (`partitionInvalidateAll`).
* Malformed input fails the load with a message, it must not terminate the application.

## Options

* Every option goes through `parameterRegistry` in `lodclusters.cpp`. The help text ends with its default and range.
* The UI widget for it uses the label the docs quote, as _"Label"_, and the same range. Clamp the command line value
  the way the UI does.
* Removing an option breaks command lines that still pass it. Say so in the CHANGELOG.

## Cache file

* Processing results are stored next to the scene (`.nvsngeo`). Large scenes write it while processing.
* A change to what is stored bumps `geoVersion` in `scene.hpp`, once per feature rather than per intermediate step,
  with a line in its history comment and a **WARNING** in the CHANGELOG entry. Check for collisions with other
  pending changes that bump it as well.

## CHANGELOG

* Newest dated entry on top, one bullet per user visible change, a link to the doc of a new technique.
* Terse: what changed and the one reason that matters, no measurement tables.
* Fixes of features that were never released get no bullet.

## Documentation

* A technique gets `docs/<name>.md` with *Motivation*, *Algorithm*, *Results and limitations* and *Implementation*,
  and a short README section that links it.
* Quote UI labels as _"Label"_ and give the option with its default: _"Early out"_ (`--partitionearlyout 1`, on by
  default).
* Every statement has to match the code. Re-read the doc after a change of behavior, defaults or names.
* Results state the GPU, the settings and how they were measured. Credits go into an *Acknowledgements* section.
* Figures are SVG on a white background in the style of the existing ones, with the font stack
  `'NVIDIA Sans', 'NVIDIA Sans Regular', Arial, sans-serif`.
* Only the public scenes are named, in docs and comments: Zorah, Caldera, junkworld and the bundled
  bunny (grid).

## Performance measurements

* Lock the GPU clocks.
* Compare configurations inside one process, interleaved and repeated (A B B A ...), each from the same starting state.
  Frame times of separate runs are not comparable.
* Let streaming settle before the measured frames, average over a fixed window (`--sequenceaverages`).
* Report what the measurement shows, nothing inferred.

## Tests

* `tests/run_tests.sh` (or `run_tests.ps1`) runs every `tests/seq_*.txt`, a `.cfg` of the same name picks its scene.
  It checks that every permutation builds and renders: exit code, log errors, one screenshot per sequence.
* A new option or code path gets sequences that reach it, `tests/README.md` keeps the counts.
* Run the suite with the validation layers before handing in a change to the renderers.
