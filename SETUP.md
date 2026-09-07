# Environment setup

Supported host: x86-64 Linux. Run all commands from the repository root.
Internet access is required to fetch the ARM toolchain, QEMU, and Python
packages.

## 1. Host requirements

Install:

- Python 3.11–3.12, including development headers and `venv` support
- GCC/G++ 10–14
- `make`, `m4`, `pkg-config`, `wget`, `tar`, `xz`, and `sha256sum`
- zlib development headers

On Debian/Ubuntu, the additional packages are typically `python3-dev`,
`python3-venv`, and `zlib1g-dev`. On RHEL-compatible systems they are
typically `python3-devel` and `zlib-devel`.

## 2. Download the ARM toolchain and QEMU

```bash
bash scripts/setup_tools.sh
```

The script installs hash-verified ARM GNU Toolchain 15.2.rel1 and QEMU 7.2.0
under `tools/`; no system directory is modified.

## 3. Install Python dependencies

```bash
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else "ERROR: CacheFlex requires Python 3.11 or 3.12")'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The requirements file pins the complete plotting environment. In particular,
Matplotlib 3.11.0 matches the renderer used for the supplied paper figures.

## 4. Load the environment

```bash
source setup_env.sh
```

Run this command again in every new shell. It activates the repository-local
Python tools and defines `CACHEFLEX_ROOT`, `CROSS_CXX`, `QEMU`, `GEM5`,
`SPM_COMPILER`, and `M5OP_OBJ`.

## 5. Build gem5

```bash
cd gem5
scons build/ARM/gem5.opt -j4
cd ..
```

## 6. Build the gem5 ROI-marker object

```bash
cd gem5/util/m5
scons build/arm64/abi/arm64/m5op.o \
  arm64.CROSS_COMPILE=aarch64-none-linux-gnu-
mkdir -p build/arm64/out
cp build/arm64/abi/arm64/m5op.o build/arm64/out/
cd ../../..
```

Reload the environment after both builds:

```bash
source setup_env.sh
```

## 7. Smoke tests

Build the small GEMM kernel used by the gem5 smoke test, then run both tests:

```bash
bash tests/smoke_qemu.sh
bash kernels/gemm/cacheflex/build.sh
bash tests/smoke_gem5.sh
```

Both tests must pass before running the paper experiments. Continue with
[README.md](README.md).
