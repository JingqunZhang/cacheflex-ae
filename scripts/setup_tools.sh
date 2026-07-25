#!/bin/bash
# Fetch the pinned ARM cross compiler and qemu-aarch64-static into ./tools/.
# Downloads use fixed .part names, verified SHA256 digests, and atomic installs.
# No sudo is needed and no system directory is modified.

set -euo pipefail

if [ -z "${CACHEFLEX_ROOT:-}" ]; then
    CACHEFLEX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
TOOLS_DIR="$CACHEFLEX_ROOT/tools"
mkdir -p "$TOOLS_DIR"

# The same interpreter is used for SCons and the plotting environment.
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required (Python 3.11 or 3.12)." >&2
    exit 2
fi
if ! python3 -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else 1)'
then
    echo "ERROR: CacheFlex requires Python 3.11 or 3.12; found $(python3 --version 2>&1)." >&2
    exit 2
fi

for command_name in wget sha256sum tar mktemp; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required host command is missing: $command_name" >&2
        exit 2
    fi
done

download_verified() {
    local url="$1"
    local part="$2"
    local expected_sha="$3"
    local label="$4"

    # -O overwrites an interrupted file instead of making a stale .1 sibling.
    rm -f -- "$part"
    wget --progress=bar:force -O "$part" "$url"
    if ! printf '%s  %s\n' "$expected_sha" "$part" | sha256sum -c -; then
        rm -f -- "$part"
        echo "ERROR: SHA256 verification failed for $label." >&2
        exit 4
    fi
}

quarantine_invalid() {
    local path="$1"
    local backup="${path}.invalid.$$"
    if [ -e "$path" ] || [ -L "$path" ]; then
        if [ -e "$backup" ] || [ -L "$backup" ]; then
            backup="${backup}.${RANDOM}"
        fi
        mv -- "$path" "$backup"
        echo "      Existing invalid installation moved to: $backup" >&2
    fi
}

# ARM GNU Toolchain.
ARM_TC_VER="15.2.rel1"
ARM_TC_NAME="arm-gnu-toolchain-${ARM_TC_VER}-x86_64-aarch64-none-linux-gnu"
ARM_TC_TAR="${ARM_TC_NAME}.tar.xz"
ARM_TC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/${ARM_TC_VER}/binrel/${ARM_TC_TAR}"
ARM_TC_SHA256="9a685b335bd709d683a8c782253c37e8c36c10e6924e59e39d4769b02132eb43"
ARM_TC_DIR="$TOOLS_DIR/$ARM_TC_NAME"
ARM_CXX="$ARM_TC_DIR/bin/aarch64-none-linux-gnu-g++"
ARM_PART="$TOOLS_DIR/${ARM_TC_TAR}.part"
ARM_STAGE=""

verify_arm_compiler() {
    local compiler="$1"
    local version
    [ -x "$compiler" ] || return 1
    "$compiler" --version >/dev/null 2>&1 || return 1
    version=$("$compiler" -dumpfullversion -dumpversion 2>/dev/null) || return 1
    case "$version" in
        15.2*) return 0 ;;
        *) return 1 ;;
    esac
}

cleanup_stage() {
    if [ -n "$ARM_STAGE" ] && [ -d "$ARM_STAGE" ]; then
        rm -rf -- "$ARM_STAGE"
    fi
}
trap cleanup_stage EXIT

if verify_arm_compiler "$ARM_CXX"; then
    echo "[1/2] ARM GNU Toolchain ${ARM_TC_VER} verified, skipping download"
else
    quarantine_invalid "$ARM_TC_DIR"
    echo "[1/2] Downloading ARM GNU Toolchain (~152 MiB)..."
    download_verified "$ARM_TC_URL" "$ARM_PART" "$ARM_TC_SHA256" "$ARM_TC_TAR"
    echo "      Extracting..."
    ARM_STAGE=$(mktemp -d "$TOOLS_DIR/.arm-toolchain.XXXXXX")
    tar xf "$ARM_PART" -C "$ARM_STAGE"
    rm -f -- "$ARM_PART"
    STAGED_ARM_CXX="$ARM_STAGE/$ARM_TC_NAME/bin/aarch64-none-linux-gnu-g++"
    verify_arm_compiler "$STAGED_ARM_CXX" || {
        echo "ERROR: verified archive did not contain ARM GNU Toolchain 15.2." >&2
        exit 4
    }
    # The staging directory and destination share a filesystem, so this
    # directory rename exposes only a complete toolchain.
    mv -- "$ARM_STAGE/$ARM_TC_NAME" "$ARM_TC_DIR"
    rmdir -- "$ARM_STAGE"
    ARM_STAGE=""
    verify_arm_compiler "$ARM_CXX" || {
        echo "ERROR: installed ARM compiler failed its version check." >&2
        exit 4
    }
    echo "      ARM toolchain ready: $ARM_TC_DIR/bin/"
fi

# Pinned QEMU user emulator for tests/smoke_qemu.sh.
QEMU_RELEASE="v7.2.0-1"
QEMU_URL="https://github.com/multiarch/qemu-user-static/releases/download/${QEMU_RELEASE}/qemu-aarch64-static"
QEMU_SHA256="dce64b2dc6b005485c7aa735a7ea39cb0006bf7e5badc28b324b2cd0c73d883f"
QEMU_BIN="$TOOLS_DIR/usr/bin/qemu-aarch64-static"
QEMU_PART="${QEMU_BIN}.part"

verify_qemu_binary() {
    local binary="$1"
    local actual_sha
    [ -x "$binary" ] || return 1
    actual_sha=$(sha256sum "$binary") || return 1
    actual_sha=${actual_sha%% *}
    [ "$actual_sha" = "$QEMU_SHA256" ] || return 1
    "$binary" --version 2>/dev/null |
        grep -q '^qemu-aarch64 version 7\.2\.0'
}

if verify_qemu_binary "$QEMU_BIN"; then
    echo "[2/2] QEMU 7.2.0 verified, skipping download"
else
    quarantine_invalid "$QEMU_BIN"
    mkdir -p "$(dirname "$QEMU_BIN")"
    echo "[2/2] Downloading pinned qemu-aarch64-static (~10 MiB)..."
    download_verified "$QEMU_URL" "$QEMU_PART" "$QEMU_SHA256" \
        "qemu-aarch64-static ${QEMU_RELEASE}"
    chmod +x "$QEMU_PART"
    verify_qemu_binary "$QEMU_PART" || {
        rm -f -- "$QEMU_PART"
        echo "ERROR: downloaded QEMU failed its version check." >&2
        exit 4
    }
    # Atomic replacement prevents an interrupted install from exposing a
    # partial executable.
    mv -f -- "$QEMU_PART" "$QEMU_BIN"
    verify_qemu_binary "$QEMU_BIN" || {
        echo "ERROR: installed QEMU failed its version check." >&2
        exit 4
    }
    echo "      QEMU ready: $QEMU_BIN"
fi

echo ""
echo "=== setup_tools.sh complete ==="
echo "  Cross compiler : $ARM_CXX"
echo "  QEMU           : $QEMU_BIN (7.2.0, pinned static binary)"
echo ""
echo "Next steps:"
echo "  source $CACHEFLEX_ROOT/setup_env.sh"
echo "  bash $CACHEFLEX_ROOT/tests/smoke_qemu.sh"
