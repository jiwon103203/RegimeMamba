# Running Regime Mamba on Windows

Regime Mamba is built on [`mamba-ssm`](https://github.com/state-spaces/mamba) and
[`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d). Both packages ship custom
CUDA kernels and officially support **Linux + NVIDIA GPU** only. This guide explains why
Windows needs extra work and walks through the ways to run this project on a Windows PC.

> **TL;DR** — Use **WSL2 (Option A)**. It runs the unmodified Linux toolchain on your
> Windows GPU and only takes a few commands. Build natively on Windows (Option C) only if
> you cannot use WSL2.
>
> **Scope.** This guide covers the Mamba part of the repository (`regime_mamba/`,
> `scripts/`). The statistical Jump Model pipeline (`regime_jm/`, `run_pipeline.py`) does
> not use torch or `mamba-ssm`, so it runs natively on Windows (no GPU needed) after
> `pip install -r requirements-jm.txt`.

---

## 1. Why Windows needs extra work

| Component | What it needs | Windows situation |
|-----------|---------------|-------------------|
| `mamba_ssm.Mamba` (used in `regime_mamba/models/mamba_model.py`, `e2e_regime_mamba.py`, `rl_regime_mamba.py`) | Compiled CUDA extension `selective_scan_cuda` | Official releases only publish Linux wheels, so on Windows you have to compile it yourself |
| `causal_conv1d` | Compiled CUDA extension | Same: Linux wheels only |
| `import mamba_ssm` | [Triton](https://github.com/triton-lang/triton) (imported by `mamba_ssm/ops/triton/*`) | The official `triton` package has no Windows wheels. Use the community port [`triton-windows`](https://github.com/triton-lang/triton-windows) |
| `Block(..., fused_add_norm=True)` in this repo | Triton fused LayerNorm kernel | Needs a working Triton (see above) |
| Selective-scan kernels | NVIDIA GPU | There is **no CPU fallback** on any OS, so `--gpu_id -1` cannot run the Mamba models |
| `jumpmodels` plotting (imported by `regime_mamba/models/jump_model.py`) | Sets `text.usetex=True` on import, so it needs a LaTeX install | `setup_and_run.sh` installs TeX Live with `apt-get`, which doesn't exist on Windows |
| `setup_and_run.sh` | bash, `apt-get`, `sudo` | Not available in PowerShell/cmd |

Two more details make a native install harder:

* The PyPI source tarballs of some versions (for example `mamba-ssm==2.2.4` and
  `causal-conv1d==1.5.0.post8`) **do not include the `csrc/` CUDA sources**. On Linux,
  `pip` downloads a prebuilt wheel from GitHub Releases, so nobody notices. On Windows no
  prebuilt wheel exists, so the build fails with errors like
  `Cannot open source file: 'csrc/selective_scan/selective_scan.cpp'`
  ([state-spaces/mamba#662](https://github.com/state-spaces/mamba/issues/662)).
  **On Windows, build from a `git clone` of the matching tag, not from PyPI.**
* `mamba-ssm>=2.3.2` adds more GPU-compiler dependencies (`tilelang`, `quack-kernels`,
  `triton>=3.5`) that are not needed by this project (it only uses Mamba-1). On native
  Windows, pin `mamba-ssm` to **2.2.x (for example 2.2.4)**.

---

## 2. Choosing an option

| Option | Difficulty | GPU training | Code changes | Recommended for |
|--------|-----------|--------------|--------------|-----------------|
| **A. WSL2 + Ubuntu** | Easy | ✅ (NVIDIA driver on Windows) | None | Almost everyone |
| **B. Docker Desktop (WSL2 backend)** | Easy–medium | ✅ | None | Reproducible or shared environments |
| **C. Native Windows build** | Hard | ✅ (NVIDIA, see triton-windows GPU list) | None (patches the `mamba-ssm` sources) | Users who cannot use WSL2 |
| **D. Remote Linux (lab server, cloud VM, Colab)** | Easy | ✅ | None | No local NVIDIA GPU |

Reference versions used below. They are known to have prebuilt Linux wheels for Python
3.10–3.12 and match this repository's API usage:

| Package | Version |
|---------|---------|
| Python | 3.10 – 3.12 |
| PyTorch | 2.5.1 (CUDA 12.4 build, `cu124`) |
| causal-conv1d | 1.5.0.post8 |
| mamba-ssm | 2.2.4 |

Newer versions generally work too. Just keep PyTorch, CUDA, `causal-conv1d` and
`mamba-ssm` consistent with each other.

---

## 3. Option A — WSL2 (recommended)

WSL2 runs a real Linux kernel. NVIDIA's Windows driver exposes the GPU to it, so the
Linux instructions for this repository work unchanged.

### 3.1 Prerequisites

* Windows 10 21H2+ or Windows 11
* An NVIDIA GPU with a recent Windows driver (Game Ready or Studio) that supports WSL.
  **Install the driver on Windows only. Do not install a Linux NVIDIA driver inside WSL.**

### 3.2 Install WSL2 and Ubuntu

In an **administrator PowerShell**:

```powershell
wsl --install -d Ubuntu-22.04
# reboot when prompted, then create your Linux user
wsl --update
wsl -l -v          # VERSION column must show 2
```

### 3.3 Check GPU access (inside Ubuntu)

```bash
nvidia-smi         # should list your GPU
```

If `nvidia-smi` is not found or shows no GPU, update the Windows NVIDIA driver and run
`wsl --shutdown` from PowerShell. Then open Ubuntu again.

### 3.4 CUDA toolkit inside WSL

The reference versions download prebuilt wheels, so nothing gets compiled. But the
`setup.py` of `mamba-ssm`/`causal-conv1d` still runs `nvcc` to detect the CUDA version.
Without `nvcc` the install fails with
`NameError: name 'bare_metal_version' is not defined`. So install a CUDA toolkit whose
major version matches PyTorch (for example 12.4 for `cu124`). Use NVIDIA's **WSL-Ubuntu**
repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

> Install only a `cuda-toolkit-12-x` package. **Do not** install the `cuda`, `cuda-12-x`
> or `cuda-drivers` meta-packages in WSL, because they try to install a Linux NVIDIA
> driver and break GPU access.

### 3.5 Install the project

Clone the project **into the Linux filesystem** (for example `~/`), not under `/mnt/c/...`.
Accessing Windows drives from WSL is much slower.

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-dev build-essential

git clone <this-repository-url> ~/RegimeMamba
cd ~/RegimeMamba

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1) PyTorch with CUDA (pick the command for your setup at https://pytorch.org)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 2) Build helpers must be installed first because of --no-build-isolation
pip install packaging ninja wheel setuptools

# 3) CUDA extensions (needs nvcc from 3.4; prebuilt Linux wheels are then downloaded)
pip install causal-conv1d==1.5.0.post8 --no-build-isolation
pip install mamba-ssm==2.2.4 --no-build-isolation

# 4) This package and its remaining dependencies (jumpmodels, pyyaml, ...)
pip install -e .

# 5) LaTeX, needed by jumpmodels' matplotlib settings (see section 8)
sudo apt-get install -y texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended dvipng cm-super
```

Then run the checks in [section 7](#7-verify-the-installation).

### 3.6 WSL tips

* **Memory**: WSL2 gets only part of the host RAM by default. If training or a source
  build gets killed, raise the limit in `%UserProfile%\.wslconfig` and run `wsl --shutdown`:

  ```ini
  [wsl2]
  memory=16GB
  processors=8
  swap=8GB
  ```

  When compiling from source, also set `export MAX_JOBS=4` to cap parallel `nvcc` jobs.
* **Editors**: VS Code's *WSL* extension (`code .` from the Ubuntu shell) or PyCharm's
  WSL interpreter work directly on the Linux filesystem.
* **Files**: Windows Explorer can open the WSL files at `\\wsl$\Ubuntu-22.04\home\<user>`.
  Results written to `./train_backtest_results` show up there.

---

## 4. Option B — Docker Desktop

Docker Desktop with the **WSL2 backend** passes the NVIDIA GPU through to Linux containers
(`--gpus all`). Complete [3.1](#31-prerequisites) first, then install Docker Desktop and
enable *Settings → General → Use the WSL 2 based engine*.

Example `Dockerfile` (put it in the repository root). Use a `-devel` image because it
includes `nvcc`, which the `mamba-ssm`/`causal-conv1d` installers need (see
[3.4](#34-cuda-toolkit-inside-wsl)):

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/RegimeMamba
COPY . .

RUN pip install --no-cache-dir packaging ninja wheel setuptools \
 && pip install --no-cache-dir causal-conv1d==1.5.0.post8 --no-build-isolation \
 && pip install --no-cache-dir mamba-ssm==2.2.4 --no-build-isolation \
 && pip install --no-cache-dir -e .
```

Build and run it from PowerShell:

```powershell
docker build -t regime-mamba .
docker run --rm -it --gpus all --shm-size=8g `
  -v ${PWD}/train_backtest_results:/workspace/RegimeMamba/train_backtest_results `
  regime-mamba `
  python scripts/rolling_window_train_backtest.py --config regime_mamba/config/paper_config.yaml
```

`--shm-size` matters because the `DataLoader`s in this repository use worker processes
(`num_workers=2`–`4`), and the default container shared memory is small.

---

## 5. Option C — Native Windows build (advanced)

> ⚠️ Native Windows is **not officially supported** by `mamba-ssm`/`causal-conv1d`.
> The steps below follow community reports and may need changes for your
> Python/PyTorch/CUDA combination. If anything goes wrong, fall back to Option A.

### 5.1 Prerequisites

1. **NVIDIA GPU** supported by `triton-windows` (RTX 30xx/40xx/50xx is the smoothest; see
   the [triton-windows README](https://github.com/triton-lang/triton-windows) for the
   current list).
2. **Visual Studio 2022 Build Tools** with the *Desktop development with C++* workload
   (MSVC v143 + Windows 10/11 SDK).
3. **CUDA Toolkit 12.x** whose major version matches your PyTorch build (for example
   CUDA 12.4 for `torch==2.5.1+cu124`). `nvcc --version` must work.
4. **Python 3.10–3.12 (64-bit)** and **Git**.
5. Latest **Microsoft Visual C++ Redistributable**.
6. **Long path support**. The CUDA build creates deeply nested temp paths:

   ```powershell
   # administrator PowerShell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
     -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   git config --system core.longpaths true
   ```

### 5.2 Match PyTorch and triton-windows

`triton-windows` must match your PyTorch version (from its README):

| PyTorch | triton-windows | pip spec |
|---------|----------------|----------|
| 2.5 | 3.1 | `"triton-windows<3.2"` |
| 2.6 | 3.2 | `"triton-windows<3.3"` |
| 2.7 | 3.3 | `"triton-windows<3.4"` |
| 2.8 | 3.4 | `"triton-windows<3.5"` |
| 2.9 – 2.11 | 3.5 – 3.6 | `"triton-windows<3.7"` |

### 5.3 Build steps

Open **"x64 Native Tools Command Prompt for VS 2022"**. It must be x64; the plain
Developer Command Prompt targets x86. Then run:

```bat
cd %USERPROFILE%\RegimeMamba
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip

:: 1) PyTorch (CUDA build) + matching Triton
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install "triton-windows<3.2"
pip install packaging ninja wheel setuptools einops transformers

:: 2) Build environment
set DISTUTILS_USE_SDK=1
set MAX_JOBS=4
set CAUSAL_CONV1D_FORCE_BUILD=TRUE
set MAMBA_FORCE_BUILD=TRUE

:: 3) causal-conv1d, from the git tag (PyPI sdist lacks csrc/)
cd %USERPROFILE%
git clone --depth 1 --branch v1.5.0.post8 https://github.com/Dao-AILab/causal-conv1d
cd causal-conv1d
pip install . --no-build-isolation -v

:: 4) mamba-ssm, from the git tag (apply the patch in 5.4 first!)
cd %USERPROFILE%
git clone --depth 1 --branch v2.2.4 https://github.com/state-spaces/mamba
cd mamba
pip install . --no-build-isolation -v

:: 5) This project (the locally built mamba-ssm/causal-conv1d satisfy its requirements)
cd %USERPROFILE%\RegimeMamba
pip install -e .
```

What the variables do:

* `DISTUTILS_USE_SDK=1`: tells PyTorch's extension builder to use the MSVC environment
  that is already active. Without it, the build aborts with a warning about the VC
  environment being activated more than once.
* `MAX_JOBS=4`: caps parallel `nvcc` jobs. Each job can use several GB of RAM.
* `*_FORCE_BUILD=TRUE`: skips the (always failing) search for a prebuilt Windows wheel.

The `mamba-ssm` build compiles kernels for several GPU architectures and can take
**30 minutes to over an hour**. MSVC warnings such as
`D9002: ignoring unknown option '-O3'` are harmless.

### 5.4 Source patches for MSVC

**`static_switch.h` (needed).** `mamba-ssm` (checked for v2.2.4 through `main`) declares
plain `constexpr` variables inside the `BOOL_SWITCH` lambdas. MSVC does not let the
nested kernel-launch lambdas use them, so compile errors point at `BOOL_SWITCH` /
`static_switch.h`. Before step 4, edit
`mamba\csrc\selective_scan\static_switch.h` and change both lines:

```diff
-            constexpr bool CONST_NAME = true;                                        \
+            static constexpr bool CONST_NAME = true;                                 \
 ...
-            constexpr bool CONST_NAME = false;                                       \
+            static constexpr bool CONST_NAME = false;                                \
```

`causal-conv1d` already uses `static constexpr` upstream, so it needs no change.

**`M_LOG2E` (if needed).** If the build fails with `identifier "M_LOG2E" is undefined`
(in `selective_scan_fwd_kernel.cuh` / `selective_scan_bwd_kernel.cuh`), replace `M_LOG2E`
with the literal `1.4426950408889634f` in those two files. MSVC defines `M_LOG2E` only
when `_USE_MATH_DEFINES` is set.

### 5.5 Prebuilt community wheels

Some community members publish Windows wheels of `causal-conv1d`/`mamba-ssm`. One example
is linked from [Dao-AILab/causal-conv1d#46](https://github.com/Dao-AILab/causal-conv1d/pull/46),
built for Python 3.10 / PyTorch 2.5.1 / CUDA 12.4. A wheel only works with the **exact**
Python/PyTorch/CUDA versions it was built against. For example, that PR thread reports DLL
load errors with PyTorch 2.6. These wheels are unofficial binaries, so only install them
from sources you trust.

---

## 6. Option D — Remote Linux

If you have no local NVIDIA GPU, use any Linux machine with one: a lab server, a cloud VM
(AWS/GCP/Azure GPU instance), or Google Colab. Follow the steps in
[3.5](#35-install-the-project) there. On Colab, prefix shell commands with `!` and skip
the `venv`. VS Code *Remote – SSH* gives you a local editing experience on Windows.

---

## 7. Verify the installation

Run this on any option. It builds the same model as this repository, runs a forward and
backward pass on the GPU, and fails immediately if the CUDA or Triton kernels are missing.

```bash
python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())

import mamba_ssm, causal_conv1d
print("mamba_ssm", mamba_ssm.__version__, "| causal_conv1d", causal_conv1d.__version__)

from regime_mamba.models.mamba_model import TimeSeriesMamba
model = TimeSeriesMamba(input_dim=4, d_model=8, d_state=32, n_layers=4).cuda()
x = torch.randn(2, 60, 4, device="cuda")
y = model(x)
y.sum().backward()
print("forward/backward OK:", tuple(y.shape))
EOF
```

On native Windows (cmd), save the Python part to `check_install.py` and run
`python check_install.py`.

Then start a run, for example:

```bash
python scripts/rolling_window_train_backtest.py --config regime_mamba/config/paper_config.yaml
```

---

## 8. Windows-specific runtime notes for this repository

* **LaTeX for plots.** `regime_mamba/models/jump_model.py` (the Mamba + Jump Model path,
  e.g. `--jump_model True`) imports `jumpmodels.plot`, which sets `text.usetex=True`
  globally. Saving those figures therefore needs LaTeX, `dvipng` and the Computer Modern
  fonts. The `regime_jm` pipeline does not import `jumpmodels.plot` and needs no LaTeX.
  * WSL / Docker / Linux: install the TeX Live packages listed in
    [3.5](#35-install-the-project).
  * Native Windows: install [MiKTeX](https://miktex.org/) (allow on-the-fly package
    installation) and make sure `latex` and `dvipng` are on `PATH`.
  * To skip LaTeX entirely, turn it off **after** the project modules are imported:

    ```python
    import matplotlib.pyplot as plt
    from regime_mamba.models import jump_model  # jumpmodels sets usetex=True here
    plt.rcParams["text.usetex"] = False
    ```

* **Multiprocessing.** Windows starts `DataLoader` workers (`num_workers=2`–`4` in this
  repo) and `ProcessPoolExecutor` workers by *spawning* new interpreters. The provided
  entry points (`scripts/*.py`, `run_pipeline.py`) already have an
  `if __name__ == "__main__":` guard. Keep that guard in your own
  entry scripts, and keep everything the workers need (datasets, functions) in importable
  modules rather than in notebook cells or `__main__`.
* **`setup_and_run.sh`** is a bash script. Use it as-is under WSL/Docker. On native
  Windows, follow [5.3](#53-build-steps) instead.
* **CPU-only machines** cannot run the Mamba models (`--gpu_id -1`) on any OS, because
  `mamba-ssm` has no CPU kernels. Pure-PyTorch ports such as
  [`mambapy`](https://github.com/alxndrTL/mamba.py) run on CPU. However, they have a
  different API and do not provide `mamba_ssm.modules.block.Block`, so using one would
  require changes to `regime_mamba/models/*`. Results and checkpoints would also not be
  identical to the paper setup.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'triton'` when importing `mamba_ssm` | Triton missing (native Windows) | `pip install "triton-windows<X.Y"` matching your PyTorch ([5.2](#52-match-pytorch-and-triton-windows)) |
| `ModuleNotFoundError: No module named 'selective_scan_cuda'` | CUDA extension was not built or does not match the installed PyTorch | Reinstall `mamba-ssm` with `--no-build-isolation` after installing PyTorch; rebuild if you upgraded PyTorch |
| `ImportError: DLL load failed` (Windows) | Wheel or extension built for a different PyTorch/CUDA | Rebuild from source against the installed PyTorch, or install the PyTorch version the wheel was built for |
| `Cannot open source file: 'csrc/selective_scan/...'` | Building from a PyPI sdist that lacks `csrc/` | Build from a `git clone` of the tag ([5.3](#53-build-steps)) |
| `NameError: name 'bare_metal_version' is not defined`, `nvcc was not found`, `CUDA_HOME` errors | CUDA toolkit missing or not on `PATH` | Install the CUDA toolkit ([3.4](#34-cuda-toolkit-inside-wsl); WSL: "WSL-Ubuntu" variant) and set `CUDA_HOME` |
| `The detected CUDA version mismatches the version that was used to compile PyTorch` | Toolkit major version ≠ PyTorch CUDA major version | Install a matching toolkit or a matching PyTorch build |
| Build killed / `cl.exe` or `nvcc` out of memory | Too many parallel compile jobs | `MAX_JOBS=2`–`4`; on WSL also raise `memory` in `.wslconfig` |
| `nvidia-smi` fails inside WSL | Old Windows driver or a Linux driver installed inside WSL | Update the Windows driver, remove Linux `nvidia-*` driver packages inside WSL, `wsl --shutdown` |
| `RuntimeError: latex was not able to process ...` / `dvipng` not found | `jumpmodels` enabled `usetex` | Install LaTeX ([8](#8-windows-specific-runtime-notes-for-this-repository)) or set `text.usetex=False` |
| Training is slow under WSL | Project lives on `/mnt/c/...` | Move the repository into the Linux home directory |

---

## References

* mamba-ssm: <https://github.com/state-spaces/mamba> ·
  Windows install issue: <https://github.com/state-spaces/mamba/issues/662>
* causal-conv1d: <https://github.com/Dao-AILab/causal-conv1d> ·
  Windows support PR: <https://github.com/Dao-AILab/causal-conv1d/pull/46>
* triton-windows: <https://github.com/triton-lang/triton-windows>
* CUDA on WSL: <https://docs.nvidia.com/cuda/wsl-user-guide/>
* WSL installation: <https://learn.microsoft.com/windows/wsl/install>
* Docker Desktop GPU support: <https://docs.docker.com/desktop/features/gpu/>
