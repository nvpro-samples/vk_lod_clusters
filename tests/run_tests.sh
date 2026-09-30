#!/usr/bin/env bash
# Linux / msys counterpart of run_tests.ps1, see that file and README.md for the details.
#
#   ./run_tests.sh [-e exe] [-o outdir] [-f frames] [-s scene] [-c screenshotmode]
#                  [-V 0|1] [-P preset] [seq_file ...]

set -u

tests_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(dirname "$tests_dir")"

exe=""
out_dir=""
scene=""
frames=32
screenshot=2
# validation layers are on by default: without them the VUID / Validation Error scan below
# never has anything to find, which is half of what these runs are for. -V 0 for a quick pass.
validation=1
validation_preset=1

while getopts "e:o:f:s:c:V:P:" opt; do
  case "$opt" in
    e) exe="$OPTARG" ;;
    o) out_dir="$OPTARG" ;;
    f) frames="$OPTARG" ;;
    s) scene="$OPTARG" ;;
    c) screenshot="$OPTARG" ;;
    V) validation="$OPTARG" ;;
    P) validation_preset="$OPTARG" ;;
    *) echo "usage: $0 [-e exe] [-o outdir] [-f frames] [-s scene] [-c mode] [-V 0|1] [-P preset] [seq_file ...]" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [ -z "$exe" ]; then
  for candidate in "$root_dir/_bin/Release/vk_lod_clusters_internal" \
                   "$root_dir/_bin/Release/vk_lod_clusters_internal.exe" \
                   "$root_dir/_bin/vk_lod_clusters_internal"; do
    if [ -x "$candidate" ]; then exe="$candidate"; break; fi
  done
fi
if [ ! -x "$exe" ]; then
  echo "executable not found, pass -e" >&2
  exit 1
fi
# the app runs with the case directory as its working directory, so relative paths
# have to be resolved here, while they still mean something
exe="$(cd "$(dirname "$exe")" && pwd)/$(basename "$exe")"
if [ -n "$scene" ]; then
  scene="$(cd "$(dirname "$scene")" && pwd)/$(basename "$scene")"
fi

if [ -z "$out_dir" ]; then
  out_dir="$tests_dir/_results/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

sequences=("$@")
if [ ${#sequences[@]} -eq 0 ]; then
  for f in "$tests_dir"/seq_*.txt; do sequences+=("$(basename "$f")"); done
fi

# the sample keeps rendering after a pipeline fails to build, so the log has to be scanned
patterns='shaders failed|error:|ERROR:|VUID-|Validation Error|DEVICE_LOST|failed to allocate'

any_failed=0
total_elapsed=0

for name in "${sequences[@]}"; do
  sequence_file="$tests_dir/$name"
  if [ ! -f "$sequence_file" ]; then
    echo "sequence file not found: $sequence_file" >&2
    exit 1
  fi

  # a sequence file may bring its own scene: options that only take effect at startup
  # (--force16bitdispatch) or a scene shape a sequence cannot switch to on its own
  sequence_scene="$scene"
  if [ -z "$sequence_scene" ]; then
    if [ -f "${sequence_file%.txt}.cfg" ]; then
      sequence_scene="${sequence_file%.txt}.cfg"
    else
      sequence_scene="$tests_dir/bunny_grid.cfg"
    fi
  fi

  count=$(grep -c '^[[:space:]]*SEQUENCE[[:space:]]' "$sequence_file" || true)
  if [ "$count" -eq 0 ]; then
    echo "warning: $name contains no SEQUENCE, skipped" >&2
    continue
  fi

  headless_frames=$(( count * frames + frames ))
  case_dir="$out_dir/${name%.txt}"
  mkdir -p "$case_dir"

  # --sequenceframes is only registered once the sequencer has initialized, so it cannot be
  # passed on the command line: the sequence file carries it, and -f overrides it in a copy
  # that doubles as the record of what this run actually did
  run_file="$case_dir/sequence.txt"
  sed -E "s/--sequenceframes[[:space:]]+[0-9]+/--sequenceframes $frames/" "$sequence_file" > "$run_file"

  echo "running $name ($count sequences, $headless_frames frames, scene $(basename "$sequence_scene"))"

  # screenshots are written to the working directory
  started=$(date +%s)
  ( cd "$case_dir" && "$exe" \
      --scene "$sequence_scene" \
      --sequencefile "$run_file" \
      --sequencescreenshot "$screenshot"       --validation "$validation"       --validationpreset "$validation_preset" \
      --headless \
      --headlessframes "$headless_frames" ) 2>&1 | tee "$case_dir/output.log"
  exit_code=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - started ))
  total_elapsed=$(( total_elapsed + elapsed ))

  hits=$(grep -E -c "$patterns" "$case_dir/output.log" || true)
  shots=$(ls "$case_dir"/screenshot_*.jpg 2>/dev/null | wc -l)

  result=ok
  if [ "$exit_code" -ne 0 ] || [ "$hits" -gt 0 ] \
     || { [ "$screenshot" -ne 0 ] && [ "$shots" -lt "$count" ]; }; then
    result=FAIL
    any_failed=1
    grep -E -n "$patterns" "$case_dir/output.log" | head -20
  fi

  printf '%-28s %-5s sequences %-4s screenshots %-4s exit %-3s suspicious %-4s %ss\n' \
         "$name" "$result" "$count" "$shots" "$exit_code" "$hits" "$elapsed"
done

if [ "$any_failed" -ne 0 ]; then
  echo "FAILED - see the logs and screenshots under $out_dir" >&2
  exit 1
fi

echo "all sequences ran in ${total_elapsed}s, screenshots under $out_dir"
