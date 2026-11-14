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

Paths are defined with one or more `--addcamerapath` options; the index equals the order in which they were added (the first is `0`). `--loadcamerapaths "file.txt"` instead replaces the whole set with the paths from a text file (see the format below). Paths given on the command line are **global** and take precedence: when any are present the per-scene `<scene>.camerapaths.txt` file is not auto-loaded. `--runcamerapath <index> <framecount>` then spreads the selected path across exactly `<framecount>` rendered frames (frame `f` maps to path position `f / (framecount-1)`), so it is fully deterministic. `--camerapathloop 1` wraps
instead of holding at the end, so `<framecount>` sets one lap; it needs a closed path.

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

## Recording a video

_"Capture frames"_ (`--framecapture 1`, off by default, UI: _Misc Settings → Frame Capture_) writes every rendered frame to `<directory>/frame_<index>.png`, numbered from 0 for each capture, or encodes them into `<directory>/video.mp4` with ffmpeg or `<directory>/video.h264` with Vulkan Video:

* _"Directory"_ (`--framecapturedir`, default `framecapture`) is created if missing, a new capture overwrites the files of an earlier one.
* _"Whole window"_ (`--framecapturewindow 1`, off by default) captures the window with its UI, as presented, instead of the rendered viewport. A headless run has no window and captures the viewport.
* _"Format"_ (`--framecaptureformat`, default `0`): `0` png, `1` jpg, `2` video.mp4 (ffmpeg), `3` video.h264 (Vulkan Video).
* _"Writer threads"_ (`--framecapturewriters`, default `4`, 1..8) encode png or jpg files. Rendering waits for them rather than dropping frames.
* _"ffmpeg"_ (`--framecaptureffmpeg`, default `ffmpeg` from the PATH) and _"ffmpeg options"_ (`--framecaptureffmpegargs`, default `"-c:v hevc_nvenc -preset p5 -cq 20"`): the video is encoded by an ffmpeg process that reads the frames from a pipe. A compute shader converts them to nv12 (bt709, limited range) first, which more than halves what is read back and piped. Videos drop up to three columns and a row so the width is a multiple of 4 and the height even. A video keeps the size of its first frame, a resized viewport is scaled to fit with black borders. ffmpeg is not part of the sample, the options pick its encoder, e.g. `-c:v libx264 -crf 18` without an NVIDIA GPU encoder.
* _"QP"_ (`--framecaptureqp`, default `20`, 1..51): the Vulkan Video format encodes H.264 High profile on the GPU's video encode queue, without ffmpeg. An IDR picture every two seconds, P pictures in between with constant QP, written as an Annex B elementary stream; `ffmpeg -i video.h264 -c copy video.mp4` puts it into an mp4. Only the bitstream is read back, the nv12 frame is copied into the encoder's source picture and the encode queue waits for the frame's semaphore. Needs `VK_KHR_video_encode_h264`, so far only tested on NVIDIA GPUs.
* _"Real-time fps"_ (`--framecapturefps`, default `60`, 1..240) is the frame rate of the video. While capturing, **Play** advances the path by 1/fps per frame, so the frames play back at the path's speed however long they took to render. Auto exposure adapts at the same rate.

The captured image is the one a viewport screenshot saves. In a window a capture starts once the viewport has kept its size for four frames, at most 30, the UI layout settles in the first frames after startup. Camera paths and auto exposure stand still meanwhile. A windowed sequence of a fixed frame count therefore captures those few frames less. It is copied at the end of its frame and read back once its frame cycle comes around again, so capturing does not stall the GPU. Encoding is the limit. Headless runs of 240 frames of the bunny grid at 3840x2160, NVIDIA RTX PRO 5000 Blackwell and AMD Ryzen 9 7950X:

* png: about 0.23 s per file on one CPU thread, a jpg about 0.14 s. With 8 writers 22 to 28 frames per second, rendering included.
* video: ffmpeg took about 11 ms per frame with the default options. `-split_encode_mode forced` added to them spreads a frame across the GPU's encoders, about 6 ms. It needs a GPU with more than one encoder and an ffmpeg that knows the option, which is why it is not the default.
* Vulkan Video: about 6 to 7 ms per frame, the encoder runs on one of the GPU's encoders. At QP 28 the bunny grid flight was 25.5 MB for 39.1 dB luma PSNR, ffmpeg's default HEVC 16.4 MB for 39.5 dB.

For a fixed frame count, run the path with `--runcamerapath` headless, e.g. 10 s at 60 fps:

```bash
vk_lod_clusters --headless --headlessframes 600 --windowsize 3840 2160 --loadcamerapaths flythrough.camerapaths.txt --runcamerapath 0 600 --framecapture 1 --framecaptureformat 2
```

Image files are encoded afterwards, e.g. `ffmpeg -framerate 60 -i framecapture/frame_%08d.png -c:v libx265 -crf 18 -pix_fmt yuv420p flythrough.mp4`.
