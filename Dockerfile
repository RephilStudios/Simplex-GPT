# syntax=docker/dockerfile:1
# --------------------------------------------------------------------------- #
# Simplex GPT — Qwen3.5-35B-A3B thought-field endpoint, for the NVIDIA DGX Spark
#
# The BASE_IMAGE must already ship a PyTorch + CUDA build that runs on the
# Spark (aarch64, Blackwell / sm_100). We do NOT install torch here — we
# inherit it from the base so we don't clobber the GPU build.
#
# The ONE thing to confirm per-machine: pick a BASE_IMAGE tag that exists and
# supports the Spark. Override on the build (or via .env BASE_IMAGE=...):
#
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3 .
#
# The 35B-A3B weights are NOT baked in — docker-compose mounts them at
# /models/Qwen3.5-35B-A3B (see .dockerignore for the exclusion).
# --------------------------------------------------------------------------- #
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# app deps first (layer-cached); torch is already in the base image
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# project code (weights/caches are excluded via .dockerignore)
COPY . .

EXPOSE 8100

# default command (docker-compose overrides this)
CMD ["python", "serve_real_endpoint.py", \
     "--host", "0.0.0.0", "--port", "8100", \
     "--weights", "/models/Qwen3.5-35B-A3B", \
     "--model-id", "qwen3.5-35b-a3b-simplex", \
     "--device", "cuda", "--thought-enabled", \
     "--thought-seed", "42", "--gain", "2.0"]
