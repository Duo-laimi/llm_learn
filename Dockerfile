# 基础镜像建议选择包含 CUDA 的镜像，否则无法执行 nvidia-smi 和 nvcc
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 设置环境变量，确保安装过程不产生交互弹窗
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装基础工具 (Miniconda)
RUN apt-get update && apt-get install -y wget git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 将 conda 加入路径
ENV PATH=/opt/conda/bin:$PATH

# --- 步骤 1 & 2: 创建文件夹并配置 conda 路径 ---
RUN mkdir -p /workspace/envs /workspace/pkgs && \
    conda config --add envs_dirs /workspace/envs && \
    conda config --add pkgs_dirs /workspace/pkgs

# --- 步骤 3: 修改 pip 源 ---
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# --- 步骤 4: 创建 torch 环境 (Python 3.10) ---
RUN conda create -n torch python=3.10 -y

# --- 步骤 5, 6, 7, 8: 激活环境并安装库 ---
# 注意：在 Dockerfile 中，每一行 RUN 都是独立的。
# 使用 'conda run -n torch' 可以在指定环境下执行命令，无需手动 source activate。
RUN conda run -n torch nvidia-smi && \
    conda run -n torch nvcc -V

# 安装 ipykernel 并创建 kernel
RUN conda run -n torch pip install ipykernel && \
    conda run -n torch python -m ipykernel install --user --name torch --display-name "torch"

RUN export HF_HOME="/path/to/your/cache"
RUN export HF_ENDPOINT="https://hf-mirror.com"
# 安装指定的深度学习库
# 注意：flash-attn 和 unsloth 对环境依赖极高，建议分开安装以方便排查错误
RUN conda run -n torch pip install \
    torch \
    torchvision \
    transformers \
    gradio \
    dashscope \
    vllm \
    evalscope

# 安装 flash-attn 和 unsloth (通常需要编译，耗时较长)
RUN conda run -n torch pip install flash-attn --no-build-isolation
RUN conda run -n torch pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 设置工作目录
WORKDIR /workspace

# 默认启动命令：进入 bash
CMD ["/bin/bash"]