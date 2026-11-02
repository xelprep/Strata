# Local copy of upstream's source-build Dockerfile, modified for an NVIDIA Maxwell card
# (compute capability 5.2). See local/README.md for the build steps. Upstream
# changes to this file do NOT come along with git pull: after pulling, diff
# upstream's Dockerfile against this one and carry over what you need.
#
# Differences from upstream (as of the pull on 2026-10-31, v0.2.9):
#   - CUDA 12.9 base image instead of CUDA 13 (CUDA 13 cannot compile for Maxwell);
#   - CUDA_ARCHITECTURES defaults to 52 (upstream's CMake already admits sm_52 in
#     the experimental build since #3218 and sets -DSTRATA_EXPERIMENTAL_SM60=ON
#     by itself when an arch is below 75 - no CMakeLists.txt patch is needed);
#   - local/maxwell.patch is applied in BOTH stage 1 (engine sources: src/core/device.cu;
#     upstream's runtime floor is still 6.0 there without it) and stage 2 (the Terminal
#     bundle's setup.py, whose sm60_card must admit 52 or setup calls the card unsupported);
#   - BUILD_VISION defaults to 0, the way upstream's own CUDA 12 recipe does for old cards.
#
# Build (on the PC the card is in — native CPU code and the CUDA card both matter;
# the Terminal stage stops without the commit, like upstream's source builds):
#   git pull
#   cp local/Dockerfile Dockerfile
#   docker build -t strata-maxwell --build-arg CUDA_ARCHITECTURES=52 \
#              --build-arg STRATA_COMMIT=$(git rev-parse HEAD) .      # Maxwell (experimental)
#
# Run (host needs an NVIDIA driver >= 580 and nvidia-container-toolkit; upstream's
# CUDA 12 packages ask for 575.51, the Maxwell reporter's M6000 ran driver 580):
#   docker run --rm --gpus all -p 127.0.0.1:8080:8080 --ulimit memlock=-1 --shm-size 2g --stop-timeout 120 \
#              -v strata-data:/data -e MODEL=IQ3_XXS strata
#
# Setup choices are env vars, read by docker/entrypoint.sh: FAMILY, MODEL, CONTEXT,
# VISION (no | yes | cpu), KV (int8 | q4_0 | k8v4), GPU (one card) or GPUS ("0,2"
# or "all", with LAYER_SPLIT), LOW_RAM (auto | on | off), HOST, PORT, API_KEY,
# GGUF_DIR (GGUF files you already have), RESIDENT_BUDGET_GIB, KV_STREAMING.
#
# Only the model files, the prepared pack, the MTP layer and the install config
# live in the /data volume; the engine is part of the image as an engine component
# package (components/engine, packaged for sm_52). Strata loads 32-62 GB
# into RAM, so a capped container needs -e LOW_RAM=on: setup.py reads the RAM from
# /proc/meminfo, which here is the host's total, not the container's limit. Add an
# API key before exposing the port to a network: -e API_KEY=<secret>. Pass
# -e REINSTALL=1 to change the model settings later.

# CUDA 12.9: CUDA 13 dropped sm_52 (and sm_60/sm_70), so the experimental
# Maxwell / Pascal / Volta build needs a 12.x toolkit.
ARG CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
ARG RUNTIME_BASE=ubuntu:24.04

# ---- 1. compile the engine and make its component package
FROM ${CUDA_IMAGE} AS engine-build
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PATH=/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git libgomp1 libatomic1 python3 python3-pip python3-venv unzip patch \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .

# The local Maxwell changes (see local/README.md): src/core/device.cu lowers the
# experimental build's runtime floor from 6.0 to 5.2 - upstream (#3218) built the
# sm_52 floor into CMake but kept the runtime check at 6.0. Fails the build if
# upstream changed a patched file - regenerate the patch against the new checkout.
RUN patch -p1 < local/maxwell.patch

ARG CUDA_ARCHITECTURES=52
ARG BUILD_VISION=0
ARG ENGINE_VERSION=
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-build.txt
# llama.cpp at the pinned commit (ggml for the engine, mtmd for pictures), exactly as setup.py fetches it
RUN /venv/bin/python -c "import setup; print(setup.get_llama_cpp())"
RUN set -e; \
    real=""; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}; \
    last=$(echo "$CUDA_ARCHITECTURES" | tr ';,' '  ' | awk '{print $NF}'); \
    sm60=OFF; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do if [ "$a" -lt 75 ]; then sm60=ON; fi; done; \
    /venv/bin/cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_EXPERIMENTAL_SM60=$sm60 \
        -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" -DSTRATA_GGML_DIR=/src/third_party/llama.cpp; \
    /venv/bin/cmake --build build --target strata strata_kq -j "$(nproc)"; \
    if [ "$BUILD_VISION" = "1" ]; then \
      /venv/bin/cmake -G Ninja -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/src/third_party/llama.cpp \
          -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=ON -DGGML_CUDA_NCCL=OFF "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"; \
      /venv/bin/cmake --build build-vision --target strata-vision -j "$(nproc)"; \
    fi
RUN set -e; \
    mkdir -p /stage /engine; \
    cp build/strata /stage/; \
    [ -f build/libstrata_kq.so ] && cp build/libstrata_kq.so /stage/ || true; \
    [ -f build-vision/bin/strata-vision ] && cp build-vision/bin/strata-vision /stage/ || true; \
    ver="${ENGINE_VERSION:-$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)}"; \
    cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1); cudamajor=${cudafull%%.*}; \
    if [ "$cudamajor" = 13 ]; then mindriver=580.65; else mindriver=575.51; fi; \
    vision_archs=""; [ -f /stage/strata-vision ] && vision_archs=", \"vision_archs\": \"$CUDA_ARCHITECTURES\""; \
    echo "{\"version\": \"$ver\", \"source\": \"docker build\", \"archs\": \"$CUDA_ARCHITECTURES\", \"cuda\": \"$cudafull\", \"portable\": true${vision_archs}}" > /stage/BUILD.json; \
    python3 packaging/collect_libs.py --binary build/strata --search /usr/local/cuda/lib64 --out /stage --also 'libcublasLt.so*'; \
    archs=""; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do archs="$archs --arch sm_$a"; done; \
    last=$(echo "$CUDA_ARCHITECTURES" | tr ';,' '  ' | awk '{print $NF}'); \
    python3 packaging/package_engine.py --src /stage --backend cuda --version "$ver" --out /engine $archs --arch "compute_$last" \
        --min-driver nvidia=$mindriver --variant cuda$cudamajor --platform linux-x86_64 --build-id "docker-source-${ver}-cuda${cudafull}" --include SYSTEM-LIBS.txt --include 'libstrata_kq*' \
        --include strata-vision; \
    for d in /engine/*/; do python3 packaging/package_engine.py --verify "$d"; python3 packaging/binary_deps.py "$d"; done

# ---- 2. the Terminal (the release archive, built here)
FROM ${RUNTIME_BASE} AS terminal
ARG STRATA_COMMIT=
ENV STRATA_BUILD_COMMIT=${STRATA_COMMIT}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates git patch && rm -rf /var/lib/apt/lists/*
COPY . /src/
# The same patch, so the Terminal's bundled setup.py admits Maxwell cards (sm60_card
# covers 52): without this the launch path calls the card "not supported" and the
# packaged sm_52 engine never starts.
RUN cd /src && patch -p1 < local/maxwell.patch
RUN python3 /src/packaging/package_terminal.py --out /out --cache /cache \
 && mkdir -p /opt/strata \
 && tar -xzf /out/strata-terminal-*-linux-x86_64.tar.gz -C /opt/strata \
 && mv /opt/strata/strata-terminal-* /opt/strata/terminal \
 && /opt/strata/terminal/bin/strata --version
# llama.cpp at the commit setup.py pins: fetched here, so the first start does not depend on GitHub. Only gguf-py (the model preparation
# tools) and the file setup.py looks for (ggml/CMakeLists.txt) are kept: 170 MB become a few
RUN cd /opt/strata/terminal && STRATA_CONTROL_INSTALL=1 STRATA_INSTALL_DIR=/opt/strata/seed PYTHONNOUSERSITE=1 \
      /opt/strata/terminal/python/bin/python3 -c "import setup; print(setup.get_llama_cpp())" \
 && rm -rf /opt/strata/seed/third_party/*.zip* \
 && cd /opt/strata/seed/third_party/llama.cpp \
 && find . -mindepth 1 -maxdepth 1 ! -name gguf-py ! -name ggml ! -name 'LICENSE*' -exec rm -rf {} + \
 && find ggml -mindepth 1 -depth ! -type d ! -name CMakeLists.txt -delete && find ggml -mindepth 1 -type d -empty -delete \
 && du -sh /opt/strata/seed

# ---- 3. the runtime image (the same as docker/Dockerfile's, FLAVOR=cuda)
FROM ${RUNTIME_BASE}
ARG STRATA_VERSION=dev
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates libgomp1 libatomic1 libnuma1 zlib1g \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*
COPY --from=terminal /opt/strata/terminal /opt/strata/terminal
COPY --from=terminal /opt/strata/seed /opt/strata/seed
COPY --from=engine-build /engine/ /opt/strata/components/engine/
COPY docker/entrypoint.sh docker/healthcheck.sh /opt/strata/
COPY docker/strata-wrapper.sh /usr/local/bin/strata
RUN chmod 0755 /opt/strata/entrypoint.sh /opt/strata/healthcheck.sh /usr/local/bin/strata
ENV STRATA_COMPONENTS_DIR=/opt/strata/components STRATA_INSTALL_DIR=/data/install XDG_STATE_HOME=/data/state \
    XDG_CONFIG_HOME=/data/config HOME=/data/home PORT=8080 LANG=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 STRATA_IN_DOCKER=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME ["/data"]
EXPOSE 8080
LABEL org.opencontainers.image.title="Strata" com.strata.flavor="cuda" com.strata.build="from-source" com.strata.maxwell="yes" org.opencontainers.image.version="${STRATA_VERSION}"
HEALTHCHECK --interval=30s --timeout=8s --start-period=2h --retries=3 CMD ["/opt/strata/healthcheck.sh"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strata/entrypoint.sh"]
CMD ["run"]
