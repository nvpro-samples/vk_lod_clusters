# Camera Path

A camera path is a keyframed fly-through defined entirely within this sample. It can be authored in the UI, is copy/paste friendly, and can be provided on the command line just like the camera string. Its main purpose is **deterministic benchmarking**: with fixed-step playback the camera visits the exact same positions every run, independent of frame rate or GPU speed.

**UI** (_Misc Settings → Camera Paths_):

* **File:** **Save** / **Load** write and read all paths to a text file stored **next to the model file**, named after it (`<model>.camerapaths.txt`, shown under the buttons). This file is **auto-loaded** whenever that scene is loaded.
* **Path:** **New** / **Delete** add or remove a path, **Copy** / **Paste** exchange the selected path with the clipboard as a string; the dropdown below selects the active path (the index used by `--runcamerapath`).
* **Key:** **New** captures the current camera as a keyframe (inserted after the selected one); **Update** / **Delete** edit the selection. The list below shows the keyframes — selecting one previews it.
* **Smooth** uses Catmull-Rom interpolation (otherwise piecewise linear), **Loop** and **Duration (s)** affect real-time playback. When **Loop** is on and the first and last keyframes coincide, the smoothing wraps around the join for a seamless loop.
* **Play** / **Stop** / **Restart** and the **t** slider drive a real-time preview.
* **Run fixed** plays the path across **Fixed frames** frames, exactly as `--runcamerapath` does.

**Command line / benchmarking:**

Paths are defined with one or more `--addcamerapath` options; the index equals the order in which they were added (the first is `0`). `--loadcamerapaths "file.txt"` instead replaces the whole set with the paths from a text file (see the format below). Paths given on the command line are **global** and take precedence: when any are present the per-scene `<scene>.camerapaths.txt` file is not auto-loaded. `--runcamerapath <index> <framecount>` then spreads the selected path across exactly `<framecount>` rendered frames (frame `f` maps to path position `f / (framecount-1)`), so it is fully deterministic.

> Note: the per-scene file is loaded only after its scene finishes loading (which is asynchronous), so a command-line `--runcamerapath` at startup cannot reference it yet. For benchmarking either provide the path on the command line with `--addcamerapath` / `--loadcamerapaths`, or place `--runcamerapath` inside the sequences of a `--sequencefile` (which run after the scene is resident).

For benchmarking, put `--runcamerapath` in each sequence and match `--sequenceframes` to the frame count. For example a sequence script (`--sequencefile bench.txt`):

```text
SEQUENCE "flythrough raytrace"
--renderer 1
--runcamerapath 0 256

SEQUENCE "flythrough raster"
--renderer 0
--runcamerapath 0 256
```

with the path stored in a file `flythrough.camerapaths.txt`:

```text
smooth 1 loop 0 dur 10 ;
  {0, 2, 5}, {0, 0, 0}, {0, 1, 0}, {60} ;
  {5, 2, 0}, {0, 0, 0}, {0, 1, 0}, {60} ;
  {0, 2, -5}, {0, 0, 0}, {0, 1, 0}, {60}
```

run with:

```bash
vk_lod_clusters --sequenceframes 256 --loadcamerapaths flythrough.camerapaths.txt --sequencefile bench.txt
```

**String format** (also valid inside `.cfg` files as a single-line, quoted value): an optional header of `smooth <0/1> loop <0/1> dur <seconds>`, followed by `;`-separated keyframes each written as `{eye}, {center}, {up}, {fov}` (the per-keyframe fov is optional). The **Copy** button produces this canonical form. In the paths **file** a path may span any number of lines and is parsed until the next path begins (each path starts with the `smooth` header keyword); anything from a `#` to the end of a line is a comment.
