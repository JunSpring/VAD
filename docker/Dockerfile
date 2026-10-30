# VAD (VADv1 / VAD-Tiny) training image for a single RTX 4090 (sm_89).
# Derived from the GenAD image that is already verified on this host
# (/media/junspring/JunBoom/ravee-pipeline/docker/Dockerfile-genad).
# Stack: CUDA 11.8 / torch 2.0.1 / mmcv-full 1.7.1 / mmdet 2.28.2 / mmdet3d v0.17.1.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES},compute,utility

SHELL [ "/bin/bash", "--login", "-c" ]

RUN apt-get update -q && \
    apt-get install -q -y \
    wget \
    python3.10-dev \
    python3-pip \
    python3.10-tk \
    git \
    ninja-build \
    ffmpeg libsm6 libxext6 libglib2.0-0 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
RUN pip install mmdet==2.28.2 mmsegmentation==0.30.0

RUN pip install numba==0.58.1
RUN pip install yapf==0.40.1
RUN pip install numpy==1.26.4

ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV MAX_JOBS=8

# Rebuild mmcv ops for sm_89 (the prebuilt cu118 wheel is not compiled for 8.9).
RUN git clone -b v1.7.1 --single-branch https://github.com/open-mmlab/mmcv.git /tmp/mmcv && \
    cd /tmp/mmcv && \
    pip install --no-cache-dir -v -e . --no-build-isolation --no-deps && \
    cd / && rm -rf /tmp/mmcv/.git

# mmdet3d v0.17.1 with the patches needed for torch 2.x / numba 0.58.
RUN git clone -b v0.17.1 --single-branch https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdetection3d && \
    cd /tmp/mmdetection3d && \
    \
    find mmdet3d/ops -type f \( -name "*.cpp" -o -name "*.cu" \) -exec sed -i 's|#include <THC/THC.h>|#include <ATen/cuda/CUDAContext.h>|g' {} + && \
    find mmdet3d/ops -type f \( -name "*.cpp" -o -name "*.cu" \) -exec sed -i 's|AT_CHECK|TORCH_CHECK|g' {} + && \
    find mmdet3d/ops -type f \( -name "*.cpp" -o -name "*.cu" \) -exec sed -i 's|THCState_getCurrentStream(state)|at::cuda::getCurrentCUDAStream()|g' {} + && \
    find mmdet3d/ops -type f \( -name "*.cpp" -o -name "*.cu" \) -exec sed -i '/extern THCState \*state;/d' {} + && \
    \
    sed -i 's/@CONV_LAYERS.register_module()/@CONV_LAYERS.register_module(force=True)/g' mmdet3d/ops/spconv/conv.py && \
    sed -i 's/@TEST_TIME_AUGs.register_module()/@TEST_TIME_AUGs.register_module(force=True)/g' mmdet3d/models/detectors/base.py && \
    sed -i 's/from numba.errors import NumbaPerformanceWarning/from numba.core.errors import NumbaPerformanceWarning/g' mmdet3d/datasets/pipelines/data_augment_utils.py && \
    \
    pip install --no-cache-dir -v -e . --no-build-isolation --no-deps && \
    \
    cd / && rm -rf /tmp/mmdetection3d/.git

RUN cp -r /tmp/mmdetection3d/mmdet3d /usr/local/lib/python3.10/dist-packages

# mmdet3d v0.17.1 hard-asserts mmcv<=1.4.0 at import time; we run mmcv-full 1.7.1.
RUN for f in /tmp/mmdetection3d/mmdet3d/__init__.py /usr/local/lib/python3.10/dist-packages/mmdet3d/__init__.py; do \
        sed -i "s/^mmcv_maximum_version = .*/mmcv_maximum_version = '1.8.0'/" "$f"; \
    done && grep -n "mmcv_maximum_version =" /usr/local/lib/python3.10/dist-packages/mmdet3d/__init__.py

# VAD's vectorized map code mutates shapely geometries (`ext.coords = ...`) and
# relies on STRtree returning geometries -> shapely must stay < 2.0.
# nuscenes-devkit 1.1.11 is the last release pinned to Shapely<2.
RUN pip install "shapely==1.8.5" "nuscenes-devkit==1.1.11"

# VAD / mmdet3d runtime deps (mmdet3d was installed with --no-deps above).
RUN pip install timm similaritymeasures prettytable tensorboard ipython \
    lyft_dataset_sdk networkx plyfile scikit-image trimesh \
    pycocotools terminaltables pyquaternion scipy pandas tqdm PyYAML
# Re-pin after everything else so nothing drags in numpy 2.x / shapely 2.x.
RUN pip install numpy==1.26.4 "shapely==1.8.5"

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN apt-get update -q && \
    apt-get install -q -y \
    gosu \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --shell /bin/bash -u 1001 -c "" -m user && usermod -a -G dialout user
RUN echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user

# The repo is bind-mounted at the same absolute path as on the host so that
# host-side symlinks under data/ resolve inside the container.
RUN mkdir -p /home/junspring/VAD && chown -R 1001:1001 /home/junspring
WORKDIR /home/junspring/VAD

ENV PYTHONPATH=/home/junspring/VAD
# torchvision://resnet50 resolves to $TORCH_HOME/hub/checkpoints/resnet50-19c8e357.pth
ENV TORCH_HOME=/home/junspring/VAD/ckpt/torch
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["/bin/bash"]
