#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
framework_dir="$repo_dir/build/badgeware-cpp"
build_dir="$repo_dir/build/tufty"

: "${PICO_SDK_PATH:?Set PICO_SDK_PATH to a Pico SDK checkout}"
command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v arm-none-eabi-gcc >/dev/null || { echo "Arm GNU embedded toolchain is required" >&2; exit 1; }
command -v picotool >/dev/null || { echo "picotool is required" >&2; exit 1; }

if [[ ! -d "$framework_dir/.git" ]]; then
  mkdir -p "$repo_dir/build"
  git clone https://github.com/pimoroni/badgeware-cpp.git "$framework_dir"
fi
git -C "$framework_dir" checkout 7d1ef0882c6e9dcf374542528f6f211e287b5afb
git -C "$framework_dir" submodule update --init --recursive
python3 "$repo_dir/install.py" "$framework_dir"

cmake -S "$framework_dir" -B "$build_dir" -DBADGE_BOARD=tufty -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --target badge_llm -j 8
cp "$build_dir/demos/badge_llm/badge_llm.uf2" "$repo_dir/build/selectbadge-llm.uf2"
echo "Built $repo_dir/build/selectbadge-llm.uf2"
