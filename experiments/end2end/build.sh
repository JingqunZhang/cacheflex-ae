#!/bin/bash
# build.sh — compatibility wrapper.
# The end-to-end benchmark sources moved to kernels/end2end/src/;
# binaries are produced in kernels/end2end/bin/ (mirrored here via the
# experiments/end2end/bin symlink).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/../../kernels/end2end/build.sh" "$@"
if [ ! -e "$HERE/bin" ]; then
    ln -s ../../kernels/end2end/bin "$HERE/bin"
fi
