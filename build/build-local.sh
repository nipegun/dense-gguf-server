#!/bin/bash

set -euo pipefail

fCleanup() {
  :
}

trap fCleanup EXIT

fMain() {
  local vProjectRoot
  local vJobs
  local -a aGenerator=()

  vProjectRoot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 1
  vJobs="${DGS_BUILD_JOBS:-4}"
  if [[ ! "$vJobs" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'DGS_BUILD_JOBS must be a positive integer.' >&2
    return 1
  fi
  if ! command -v cmake > /dev/null 2>&1; then
    printf '%s\n' 'Install the documented build dependencies before running this script.' >&2
    return 1
  fi
  if [[ ! -f "${vProjectRoot}/_/build/CMakeCache.txt" ]] && command -v ninja > /dev/null 2>&1; then
    aGenerator=(-G Ninja)
  fi

  if ! cmake -S "${vProjectRoot}/build" -B "${vProjectRoot}/_/build" \
    "${aGenerator[@]}" -DCMAKE_BUILD_TYPE=Release "$@"; then
    return 1
  fi
  if ! cmake --build "${vProjectRoot}/_/build" --target dgs-server --parallel "$vJobs"; then
    return 1
  fi
  printf 'Built %s\n' "${vProjectRoot}/_/build/bin/dgs-server"
}

if ! fMain "$@"; then
  printf '%s\n' 'DGS build failed.' >&2
  exit 1
fi
