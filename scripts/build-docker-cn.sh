#!/usr/bin/env bash
# ============================================================================
# 1Cat-vLLM 本地 Docker 镜像构建脚本（国内网络优化版）
#
# 按照 .github/workflows/build-wheel.yml 的方式构建 docker/Dockerfile-vllm2，
# 但所有下载都走国内镜像 / 本地缓存，规避国际网络不稳定问题。
#
# 用法:
#   scripts/build-docker-cn.sh                 # 前台构建（标签=当天日期 + latest）
#   scripts/build-docker-cn.sh --background    # 后台构建（日志 /tmp/vllm-build.log）
#   scripts/build-docker-cn.sh --check         # 只检查/恢复辅助环境，不构建
#   IMAGE_TAG=v1.0 MAX_JOBS=32 scripts/build-docker-cn.sh
#
# 环境变量:
#   IMAGE_NAME     镜像名 (默认 1cat-vllm)
#   IMAGE_TAG      镜像标签 (默认 YYYYMMDD)
#   MAX_JOBS       ninja 并行度 (默认 16)
#   NVCC_THREADS   每个 nvcc 的线程数 (默认 4)
#   CUDA_VERSION   CUDA 版本 (默认 12.8.1，需与本地 nvidia/cuda 基础镜像一致)
#   TORCH_CUDA_ARCH_LIST  (默认 7.0，即 V100/SM70)
#   GH_PROXY       GitHub 加速前缀 (默认 https://ghfast.top/)
# ============================================================================
set -euo pipefail

# ---------------- 配置 ----------------
IMAGE_NAME="${IMAGE_NAME:-1cat-vllm}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d)}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
MAX_JOBS="${MAX_JOBS:-16}"
NVCC_THREADS="${NVCC_THREADS:-4}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
GH_PROXY="${GH_PROXY:-https://ghfast.top/}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIRROR_DIR="${MIRROR_DIR:-/tmp/pymirror}"        # HTTP 镜像目录
GIT_MIRROR_DIR="${GIT_MIRROR_DIR:-/tmp/gitmirror}" # git daemon 目录
DEPS_DIR="${DEPS_DIR:-/tmp/deps}"                 # CMake 依赖源码
BASE_CN_DIR="${BASE_CN_DIR:-/tmp/base-cn}"        # 基础镜像构建上下文
HTTP_PORT=8088
GIT_PORT=9411
REGISTRY_PORT=5010
BUILDER_NAME="vllm-cn"
BUILD_LOG="/tmp/vllm-build.log"

# 固定版本（与 Dockerfile 中的 pin 保持一致）
PROTOC_VERSION="34.2"
SCCACHE_VERSION="0.8.1"
FLASHINFER_VERSION="0.6.11.post2"
CUTLASS_TAG="v4.4.2"
TRITON_TAG="v3.5.1"
FA_V100_COMMIT="c2eda5e6115b98c3ba4bfd181570668742eece22"
DEEPEP_COMMIT="73b6ea4"
LMM_COMMIT="5b558989844d1c7af3e43d0f604069ffd9c06320"
PBS_TAG="20260929"   # python-build-standalone release tag

log()  { echo -e "\033[32m[build-cn]\033[0m $*"; }
warn() { echo -e "\033[33m[build-cn] WARN:\033[0m $*" >&2; }
die()  { echo -e "\033[31m[build-cn] ERROR:\033[0m $*" >&2; exit 1; }

port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }
gh() { echo "${GH_PROXY}$1"; }

# ---------------- 1. 本地 HTTP 镜像 (127.0.0.1:8088) ----------------
ensure_http_mirror_files() {
  local need=0
  # uv 二进制（版本跟随 astral.sh 安装脚本）
  local uv_ver
  uv_ver=$(curl -sL -m 30 https://astral.sh/uv/install.sh \
    | grep -oE 'astral-sh/uv/releases/download/[0-9]+\.[0-9]+\.[0-9]+' | head -1 | awk -F/ '{print $NF}')
  [ -n "$uv_ver" ] || die "无法获取 uv 版本号（astral.sh 不可达？）"
  [ -f "$MIRROR_DIR/uv/uv-x86_64-unknown-linux-gnu.tar.gz" ] || need=1
  # python-build-standalone（uv 托管 Python，stripped 变体）
  local py
  for py in "3.10.21" "3.11.16" "3.12.14" "3.13.15" "3.14.7"; do
    [ -f "$MIRROR_DIR/pbs/$PBS_TAG/cpython-${py}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz" ] || need=1
  done
  # protoc / sccache / flashinfer wheel
  [ -f "$MIRROR_DIR/protoc/protoc-${PROTOC_VERSION}-linux-x86_64.zip" ] || need=1
  [ -f "$MIRROR_DIR/sccache/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" ] || need=1
  [ -f "$MIRROR_DIR/fi/cu128/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu128-cp39-abi3-manylinux_2_28_x86_64.whl" ] || need=1
  [ -f "$MIRROR_DIR/fi/cu128/flashinfer-jit-cache/index.html" ] || need=1

  [ "$need" = 0 ] && { log "HTTP 镜像文件齐全"; return 0; }
  log "补全 HTTP 镜像文件（经 $GH_PROXY）..."
  mkdir -p "$MIRROR_DIR/uv" "$MIRROR_DIR/pbs/$PBS_TAG" "$MIRROR_DIR/protoc" "$MIRROR_DIR/sccache" "$MIRROR_DIR/fi/cu128/flashinfer-jit-cache"
  curl -sL -m 600 -o "$MIRROR_DIR/uv/uv-x86_64-unknown-linux-gnu.tar.gz" \
    "$(gh https://github.com/astral-sh/uv/releases/download/${uv_ver}/uv-x86_64-unknown-linux-gnu.tar.gz)"
  local py
  for py in "3.10.21" "3.11.16" "3.12.14" "3.13.15" "3.14.7"; do
    curl -sL -m 600 -o "$MIRROR_DIR/pbs/$PBS_TAG/cpython-${py}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz" \
      "$(gh https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${py}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz)"
  done
  curl -sL -m 300 -o "$MIRROR_DIR/protoc/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
    "$(gh https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip)"
  curl -sL -m 300 -o "$MIRROR_DIR/sccache/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    "$(gh https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz)"
  curl -sL -m 900 -o "$MIRROR_DIR/fi/cu128/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu128-cp39-abi3-manylinux_2_28_x86_64.whl" \
    "$(gh https://github.com/flashinfer-ai/flashinfer/releases/download/v${FLASHINFER_VERSION}/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu128-cp39-abi3-manylinux_2_28_x86_64.whl)"
  # flashinfer 本地 PEP 503 索引
  local sha
  sha=$(sha256sum "$MIRROR_DIR/fi/cu128/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu128-cp39-abi3-manylinux_2_28_x86_64.whl" | awk '{print $1}')
  cat > "$MIRROR_DIR/fi/cu128/flashinfer-jit-cache/index.html" <<EOF
<!DOCTYPE html>
<html><body>
<a href="/fi/cu128/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu128-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=${sha}">flashinfer_jit_cache-${FLASHINFER_VERSION}+cu128-cp39-abi3-manylinux_2_28_x86_64.whl</a>
</body></html>
EOF
  log "HTTP 镜像文件补全完成"
}

ensure_http_mirror() {
  ensure_http_mirror_files
  if port_open $HTTP_PORT; then
    log "HTTP 镜像服务已在运行 (:$HTTP_PORT)"
  else
    log "启动 HTTP 镜像服务 (:$HTTP_PORT)..."
    setsid nohup python3 -m http.server $HTTP_PORT --bind 127.0.0.1 --directory "$MIRROR_DIR" \
      > /tmp/pymirror-http.log 2>&1 < /dev/null &
    sleep 2
    port_open $HTTP_PORT || die "HTTP 镜像服务启动失败"
  fi
}

# ---------------- 2. 本地 git 镜像 (127.0.0.1:9411) ----------------
ensure_git_mirror_files() {
  if [ ! -d "$GIT_MIRROR_DIR/DeepEP.git" ]; then
    log "克隆 DeepEP@$DEEPEP_COMMIT ..."
    rm -rf "$GIT_MIRROR_DIR/DeepEP" "$GIT_MIRROR_DIR/DeepEP.git"
    git clone "$(gh https://github.com/deepseek-ai/DeepEP)" "$GIT_MIRROR_DIR/DeepEP"
    git -C "$GIT_MIRROR_DIR/DeepEP" checkout "$DEEPEP_COMMIT"
    git clone --bare "$GIT_MIRROR_DIR/DeepEP" "$GIT_MIRROR_DIR/DeepEP.git"
  fi
  if [ ! -d "$GIT_MIRROR_DIR/llm-multimodal.git" ]; then
    log "克隆 llm-multimodal@$LMM_COMMIT ..."
    rm -rf "$GIT_MIRROR_DIR/llm-multimodal" "$GIT_MIRROR_DIR/llm-multimodal.git"
    git clone "$(gh https://github.com/vllm-project/llm-multimodal)" "$GIT_MIRROR_DIR/llm-multimodal"
    # rev 可能不在任何分支上，直接按 SHA fetch 并打分支使其可达
    git -C "$GIT_MIRROR_DIR/llm-multimodal" fetch origin "$LMM_COMMIT" || \
      git -C "$GIT_MIRROR_DIR/llm-multimodal" checkout "$LMM_COMMIT"
    git -C "$GIT_MIRROR_DIR/llm-multimodal" branch pinned-lmm "$LMM_COMMIT"
    git clone --bare "$GIT_MIRROR_DIR/llm-multimodal" "$GIT_MIRROR_DIR/llm-multimodal.git"
  fi
}

ensure_git_mirror() {
  ensure_git_mirror_files
  if port_open $GIT_PORT; then
    log "git daemon 已在运行 (:$GIT_PORT)"
  else
    log "启动 git daemon (:$GIT_PORT)..."
    setsid nohup git daemon --reuseaddr --base-path="$GIT_MIRROR_DIR" --export-all \
      --port=$GIT_PORT --listen=127.0.0.1 > /tmp/git-daemon.log 2>&1 < /dev/null &
    sleep 2
    port_open $GIT_PORT || die "git daemon 启动失败"
  fi
}

# ---------------- 3. CMake 依赖源码 (/tmp/deps) ----------------
ensure_deps() {
  if [ ! -d "$DEPS_DIR/cutlass" ]; then
    log "克隆 cutlass@$CUTLASS_TAG ..."
    git clone --depth 1 --branch "$CUTLASS_TAG" "$(gh https://github.com/nvidia/cutlass.git)" "$DEPS_DIR/cutlass"
  fi
  if [ ! -d "$DEPS_DIR/triton" ]; then
    log "克隆 triton@$TRITON_TAG ..."
    git clone --depth 1 --branch "$TRITON_TAG" "$(gh https://github.com/triton-lang/triton.git)" "$DEPS_DIR/triton"
  fi
  if [ ! -d "$DEPS_DIR/fa-v100" ]; then
    log "克隆 flash-attention-v100@$FA_V100_COMMIT（含 cutlass 子模块 + SM70 补丁）..."
    git clone "$(gh https://github.com/zhinianqin/flash-attention-v100.git)" "$DEPS_DIR/fa-v100"
    git -C "$DEPS_DIR/fa-v100" checkout "$FA_V100_COMMIT"
    git -C "$DEPS_DIR/fa-v100" submodule update --init csrc/cutlass
    local p
    for p in sm70_flash_attn_d256_pipeline sm70_flash_attn_d256_splitkv3 \
             sm70_flash_attn_d256_k_pingpong sm70_flash_attn_d256_gqa_arch; do
      patch --batch --forward -p1 -l -d "$DEPS_DIR/fa-v100" -i "$REPO_ROOT/cmake/patches/$p.patch"
    done
  fi
  # 去掉 .git 减小基础镜像体积（SOURCE_DIR 方式不需要 git）
  find "$DEPS_DIR" -maxdepth 4 -name ".git" -type d -exec mv {} {}_removed \; 2>/dev/null || true
}

# ---------------- 4. 本地 registry + 自定义基础镜像 ----------------
ensure_registry() {
  if docker ps --format '{{.Names}}' | grep -q '^local-registry$'; then
    log "local registry 已在运行 (:$REGISTRY_PORT)"
  else
    log "启动 local registry (:$REGISTRY_PORT)..."
    docker rm -f local-registry 2>/dev/null || true
    docker run -d --name local-registry --restart unless-stopped \
      -p 127.0.0.1:$REGISTRY_PORT:5000 registry:2
    sleep 2
    port_open $REGISTRY_PORT || die "local registry 启动失败"
  fi
}

ensure_base_images() {
  ensure_deps
  mkdir -p "$BASE_CN_DIR"
  cp "$REPO_ROOT/scripts/cn-build/Dockerfile-devel" "$BASE_CN_DIR/Dockerfile-devel"
  cp "$REPO_ROOT/scripts/cn-build/Dockerfile-base"  "$BASE_CN_DIR/Dockerfile-base"
  # deps 通过软链接进入构建上下文
  ln -sfn "$DEPS_DIR" "$BASE_CN_DIR/deps"

  local catalog
  catalog=$(curl -s -m 5 "http://127.0.0.1:$REGISTRY_PORT/v2/_catalog" | tr -d '{}"' | tr ',' '\n' | tr -d ' ')
  local img
  for img in "devel:local/cuda-devel-cn:$CUDA_VERSION" "base:local/cuda-base-cn:$CUDA_VERSION"; do
    local kind="${img%%:*}" name="${img#*:}" reg_name="localhost:$REGISTRY_PORT/${name#local/}"
    if docker image inspect "$name" >/dev/null 2>&1; then
      log "基础镜像 $name 已存在"
    else
      log "构建基础镜像 $name ..."
      docker build -f "$BASE_CN_DIR/Dockerfile-$kind" -t "$name" "$BASE_CN_DIR"
    fi
    if echo "$catalog" | grep -q "^${name#local/}$"; then
      log "registry 中已有 ${name#local/}，跳过推送"
    else
      docker tag "$name" "$reg_name"
      docker push "$reg_name" >/dev/null
    fi
  done
}

# ---------------- 5. buildx builder ----------------
ensure_builder() {
  if docker buildx ls --format '{{.Name}}' | grep -q "^${BUILDER_NAME}$"; then
    log "buildx builder $BUILDER_NAME 已存在"
  else
    log "创建 buildx builder $BUILDER_NAME ..."
    docker buildx create --name "$BUILDER_NAME" --driver docker-container \
      --driver-opt image=docker.1ms.run/moby/buildkit:latest \
      --driver-opt network=host
  fi
  docker buildx use "$BUILDER_NAME"
}

# ---------------- 6. 构建 ----------------
run_build() {
  local scm_version
  scm_version="0.1.dev1+g$(git -C "$REPO_ROOT" rev-parse --short HEAD).d$(date +%Y%m%d)"
  log "开始构建 $IMAGE_NAME:$IMAGE_TAG (max_jobs=$MAX_JOBS, nvcc_threads=$NVCC_THREADS, arch=$TORCH_CUDA_ARCH_LIST)"

  local build_args=(
    --builder "$BUILDER_NAME"
    -f docker/Dockerfile-vllm2
    --build-arg SCCACHE_BUCKET_NAME=
    --build-arg FLASHINFER_INDEX_BASE_URL="http://127.0.0.1:$HTTP_PORT/fi"
    --build-arg "SETUPTOOLS_SCM_PRETEND_VERSION=$scm_version"
    --build-arg "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST"
    --build-arg "max_jobs=$MAX_JOBS"
    --build-arg "nvcc_threads=$NVCC_THREADS"
    --build-arg "CUDA_VERSION=$CUDA_VERSION"
    --build-arg "BUILD_BASE_IMAGE=localhost:$REGISTRY_PORT/cuda-devel-cn:$CUDA_VERSION"
    --build-arg "FINAL_BASE_IMAGE=localhost:$REGISTRY_PORT/cuda-base-cn:$CUDA_VERSION"
    --build-arg PYTORCH_CUDA_INDEX_BASE_URL=https://mirror.sjtu.edu.cn/pytorch-wheels
    -t "$IMAGE_NAME:$IMAGE_TAG"
    -t "$IMAGE_NAME:latest"
    --load
    --progress=plain
  )

  if [ "${1:-}" = "--background" ]; then
    log "后台构建，日志: $BUILD_LOG （tail -f $BUILD_LOG 查看进度）"
    ( cd "$REPO_ROOT" && setsid nohup docker buildx build "${build_args[@]}" . \
        > "$BUILD_LOG" 2>&1 < /dev/null & )
    sleep 5
    tail -3 "$BUILD_LOG" | sed 's/^/    /'
  else
    ( cd "$REPO_ROOT" && docker buildx build "${build_args[@]}" . )
  fi
}

# ---------------- 主流程 ----------------
main() {
  local mode="${1:---foreground}"
  command -v docker >/dev/null || die "未找到 docker"
  docker buildx version >/dev/null 2>&1 || die "未找到 docker buildx"

  ensure_http_mirror
  ensure_git_mirror
  ensure_registry
  ensure_base_images
  ensure_builder

  if [ "$mode" = "--check" ]; then
    log "环境检查完成，未执行构建。"
    return 0
  fi
  run_build "$mode"
  log "构建完成: $IMAGE_NAME:$IMAGE_TAG"
  docker images "$IMAGE_NAME" --format '  {{.Tag}}\t{{.Size}}\t{{.ID}}'
}

main "$@"
