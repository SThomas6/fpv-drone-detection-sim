#!/usr/bin/env bash
# Build FERS (Flexible Extensible Radar Simulator), UCT Radar Remote Sensing
# Group, GPL. Models monostatic/BISTATIC/multistatic geometries and emits
# receiver I/Q rather than a detection verdict, so cross-ambiguity processing
# runs on real samples instead of an assumed detection probability.
#
# Built inside the WSL filesystem, NOT under /mnt/c: vcpkg does tens of
# thousands of small file operations and the Windows mount makes that
# glacial, on top of git ownership warnings. Only the finished binary is
# copied back into the repo.
#
# vcpkg is cloned FULL, not --depth 1: it resolves dependency versions by
# `git show`-ing a baseline commit, which a shallow clone does not contain
# ("failed to git show versions/baseline.json").
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK=/opt/fers-build
SRC="$WORK/FERS"
VCPKG="$WORK/vcpkg"
mkdir -p "$WORK" "$REPO/third_party/bin"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake ninja-build g++ git curl zip unzip tar \
  pkg-config autoconf automake libtool bison flex python3-jinja2
command -v ninja >/dev/null || { echo "ninja STILL missing"; exit 1; }
echo "tools: cmake $(cmake --version | head -1), ninja $(ninja --version), $(g++ --version | head -1)"

[ -d "$SRC" ] || git clone --depth 1 https://github.com/stpaine/FERS "$SRC"
[ -d "$VCPKG" ] || git clone https://github.com/microsoft/vcpkg "$VCPKG"
[ -x "$VCPKG/vcpkg" ] || "$VCPKG/bootstrap-vcpkg.sh" -disableMetrics
export VCPKG_ROOT="$VCPKG"
export VCPKG_FORCE_SYSTEM_BINARIES=1

cd "$SRC"
rm -rf build && mkdir -p build && cd build
echo "=== cmake configure (vcpkg fetches and builds deps: the slow part) ==="
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG/scripts/buildsystems/vcpkg.cmake" \
  > /tmp/fers_cmake.log 2>&1 || {
    echo "cmake FAILED - tail:"; tail -40 /tmp/fers_cmake.log; exit 1; }
echo "=== build ==="
cmake --build . -j4 > /tmp/fers_make.log 2>&1 || {
    echo "build FAILED - tail:"; tail -40 /tmp/fers_make.log; exit 1; }
BIN="$(find . -type f -name 'fers*' -perm -u+x | head -1)"
[ -n "$BIN" ] || { echo "no fers binary produced"; exit 1; }
cp "$BIN" "$REPO/third_party/bin/fers"
echo "=== FERS built: $("$REPO/third_party/bin/fers" --help 2>&1 | head -2) ==="
echo "FERSDONE"
