#!/usr/bin/env bash
# TRELLIS.2: upstream checkout (pinned), conda env with Python 3.10 + the CUDA 12.4 toolkit, torch 2.6.0 cu124,
# the CUDA extensions built for this GPU, and the weights. Idempotent. Needs micromamba, git, gcc/g++ and an NVIDIA
# driver for CUDA 12.4 (>= 550). The extensions compile for the local GPU (TORCH_CUDA_ARCH_LIST, detected).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV
export PYTHONNOUSERSITE=1
E=$PWD/.conda
TRELLIS_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd

[[ -d TRELLIS.2/.git ]] || git clone -q --recursive https://github.com/microsoft/TRELLIS.2.git
git -C TRELLIS.2 checkout -q $TRELLIS_COMMIT
git -C TRELLIS.2 submodule update -q --init --recursive

[[ -x $E/bin/python ]] || micromamba create -q -y -p "$E" -c conda-forge python=3.10 git \
  cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-libraries-dev=12.4 cuda-nvrtc-dev=12.4 cuda-profiler-api=12.4 cuda-nvtx-dev=12.4
"$E/bin/pip" install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$E/bin/pip" install -q -r requirements.txt

# CUDA extensions (pinned sources in extensions/, o-voxel from the TRELLIS.2 checkout)
mkdir -p extensions
src() {  # src <dir> <url> <commit>
  [[ -d extensions/$1/.git ]] || git clone -q "$2" "extensions/$1"
  git -C "extensions/$1" checkout -q "$3"
  git -C "extensions/$1" submodule update -q --init --recursive
}
src nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4fcea7de5f396371124af597e6cc957bfae   # v0.4.0
src nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git b296927cc7fd01c2ac1087c8065c4d7248f72da4  # renderutils
src CuMesh https://github.com/JeffreyXiang/CuMesh.git 12289e1062f0603f2f0d0771b02e1395d247f26f
src FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a859c26ee8246888502eada3dd8ad85532e
[[ -d extensions/o-voxel ]] || cp -r TRELLIS.2/o-voxel extensions/o-voxel

arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
export PATH=$E/bin:$PATH CUDA_HOME=$E TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$arch} MAX_JOBS=${MAX_JOBS:-4}
export CPATH=$E/targets/x86_64-linux/include LIBRARY_PATH=$E/lib:$E/targets/x86_64-linux/lib
for pkg in nvdiffrast:nvdiffrast nvdiffrec:nvdiffrec_render CuMesh:cumesh FlexGEMM:flex_gemm o-voxel:o_voxel; do
  dir=${pkg%%:*}; dist=${pkg#*:}
  "$E/bin/pip" show -q "$dist" >/dev/null 2>&1 && continue
  echo "building $dir for compute capability $TORCH_CUDA_ARCH_LIST (several minutes)"
  "$E/bin/pip" install -q "./extensions/$dir" --no-build-isolation
done

# Weights: TRELLIS.2-4B into models/ (gen3d writes a patched pipeline config next to it), the rest into the HF cache.
# The official DINOv3 and RMBG-2.0 repos are gated: gen3d uses the ungated byte-identical DINOv3 mirror and the
# MIT BiRefNet (same architecture as RMBG-2.0) instead.
[[ -f models/TRELLIS.2-4B/pipeline.json ]] || "$E/bin/hf" download microsoft/TRELLIS.2-4B \
  --local-dir models/TRELLIS.2-4B --exclude "ckpts/*_enc_*"
"$E/bin/hf" download --quiet camenduru/dinov3-vitl16-pretrain-lvd1689m >/dev/null
"$E/bin/hf" download --quiet ZhengPeng7/BiRefNet >/dev/null
"$E/bin/hf" download --quiet microsoft/TRELLIS-image-large \
  ckpts/ss_dec_conv3d_16l8_fp16.json ckpts/ss_dec_conv3d_16l8_fp16.safetensors >/dev/null
echo "mesh: ok ($("$E/bin/python" -c 'import torch, flash_attn, cumesh, o_voxel, flex_gemm, nvdiffrast.torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'))"
