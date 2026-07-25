#!/bin/bash
# CacheFlex environment setup — source this before building/running kernels.
#   source setup_env.sh
#
# Sets CACHEFLEX_ROOT to the directory containing this script, and exports
# path variables used by build/run scripts across kernels/ and experiments/.
#
# On RHEL/CentOS 8+: if /opt/rh/gcc-toolset-11/enable exists and the host
# `as` doesn't recognise --gdwarf-5 (binutils < 2.36), source it for gem5 build:
#   source /opt/rh/gcc-toolset-11/enable
# (Not auto-sourced here so the user can pick a different toolset.)

# Resolve CACHEFLEX_ROOT to the absolute path of this script's directory.
# Works whether sourced from bash or zsh.
if [ -n "${BASH_SOURCE[0]}" ]; then
    _SETUP_SRC="${BASH_SOURCE[0]}"
else
    _SETUP_SRC="$0"
fi
export CACHEFLEX_ROOT="$(cd "$(dirname "$_SETUP_SRC")" && pwd)"
unset _SETUP_SRC

# ── Repository-local Python environment ──────────────────────────────────────
# SETUP.md creates this environment. Adding it to PATH here keeps the
# one-command-per-shell interface while avoiding system Python modifications.
export CACHEFLEX_VENV="$CACHEFLEX_ROOT/.venv"
if [ -x "$CACHEFLEX_VENV/bin/python3" ]; then
    export VIRTUAL_ENV="$CACHEFLEX_VENV"
    case ":$PATH:" in
        *":$CACHEFLEX_VENV/bin:"*) ;;
        *) export PATH="$CACHEFLEX_VENV/bin:$PATH" ;;
    esac

    # A gem5 build embeds libpython. Distribution Python installs normally
    # place that library on the system loader path, while uv/pyenv installs
    # may keep it under a private prefix. Discover that prefix from the
    # selected interpreter instead of embedding a machine-specific path.
    _CACHEFLEX_PY_LIBDIR=$(
        "$CACHEFLEX_VENV/bin/python3" -c \
            'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' \
            2>/dev/null || true
    )
    _CACHEFLEX_PY_LDLIBRARY=$(
        "$CACHEFLEX_VENV/bin/python3" -c \
            'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY") or "")' \
            2>/dev/null || true
    )
    if [ -n "$_CACHEFLEX_PY_LIBDIR" ] &&
       [ -n "$_CACHEFLEX_PY_LDLIBRARY" ] &&
       [ -f "$_CACHEFLEX_PY_LIBDIR/$_CACHEFLEX_PY_LDLIBRARY" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$_CACHEFLEX_PY_LIBDIR:"*) ;;
            *) export LD_LIBRARY_PATH="$_CACHEFLEX_PY_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
    unset _CACHEFLEX_PY_LIBDIR _CACHEFLEX_PY_LDLIBRARY
fi

# ── Cross-compile toolchain (installed by scripts/setup_tools.sh) ─────────────
export CROSS_TC="$CACHEFLEX_ROOT/tools/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu"
export CROSS_CXX="$CROSS_TC/bin/aarch64-none-linux-gnu-g++"
export CROSS_AS="$CROSS_TC/bin/aarch64-none-linux-gnu-as"
# expose the cross tools on PATH (SETUP.md invokes aarch64-none-linux-gnu-*
# by bare name, e.g. for the m5op build)
case ":$PATH:" in
    *":$CROSS_TC/bin:"*) ;;
    *) export PATH="$CROSS_TC/bin:$PATH" ;;
esac
export CROSS_LD="$CROSS_TC/bin/aarch64-none-linux-gnu-ld"

# ── QEMU user-mode static for AArch64 ─────────────────────────────────────────
export QEMU="$CACHEFLEX_ROOT/tools/usr/bin/qemu-aarch64-static"

# ── gem5 simulator binary (built by scons under gem5/) ────────────────────────
export GEM5="$CACHEFLEX_ROOT/gem5/build/ARM/gem5.opt"
export GEM5_CONFIG="$CACHEFLEX_ROOT/gem5/configs/deprecated/example/se.py"

# ── SPM compiler (encodes SPMCP / spm.ld1qd pseudo-instructions) ──────────────
export SPM_COMPILER="$CACHEFLEX_ROOT/spm_tools/spm_compiler.py"

# ── gem5 m5op helpers (built separately inside gem5/util/m5) ──────────────────
export M5_INCLUDE="$CACHEFLEX_ROOT/gem5/include"
export M5OP_OBJ="$CACHEFLEX_ROOT/gem5/util/m5/build/arm64/out/m5op.o"

# Sanity: tell the user what got set.
echo "CACHEFLEX_ROOT = $CACHEFLEX_ROOT"
[ -x "$CROSS_CXX" ]       && echo "  CROSS_CXX    : OK ($CROSS_CXX)"       || echo "  CROSS_CXX    : MISSING — run scripts/setup_tools.sh"
[ -x "$QEMU" ]            && echo "  QEMU         : OK"                     || echo "  QEMU         : MISSING — run scripts/setup_tools.sh"
[ -x "$GEM5" ]            && echo "  GEM5         : OK"                     || echo "  GEM5         : not built — cd gem5 && scons build/ARM/gem5.opt"
[ -f "$SPM_COMPILER" ]    && echo "  SPM_COMPILER : OK"                     || echo "  SPM_COMPILER : MISSING"
[ -f "$M5OP_OBJ" ]        && echo "  m5op.o       : OK"                     || echo "  m5op.o       : not built — complete SETUP.md step 6"

# Hint when on RHEL/CentOS without sourcing a modern toolset
if [ -f /opt/rh/gcc-toolset-11/enable ] && ! command -v gcc-11 >/dev/null 2>&1; then
    _gcc_major=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
    if [ -n "$_gcc_major" ] && [ "$_gcc_major" -lt 10 ]; then
        echo "  hint         : gcc is < 10; run 'source /opt/rh/gcc-toolset-11/enable' before building gem5"
    fi
    unset _gcc_major
fi
