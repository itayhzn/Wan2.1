
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

SHELL [ "bash", "-c" ]

#-------------------------------------------------
# 1) System packages
#-------------------------------------------------
RUN apt update && \
    apt install -yq \
        ffmpeg \
        build-essential \
        curl \
        wget \
        git

#-------------------------------------------------
# 2) User setup
#-------------------------------------------------
# USER vscode

# Git LFS
# RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
#     sudo apt-get install -yq git-lfs && \
#     git lfs install

#-------------------------------------------------
# 3) Miniconda installation
#-------------------------------------------------
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

#-------------------------------------------------
# 4) Copy environment file and create environment
#-------------------------------------------------
WORKDIR /storage/itaytuviah/Wan2.1

COPY environment.yml /tmp/environment.yml

RUN conda env remove -n myenv -y || true && \
    conda env create -f /tmp/environment.yml -y && \
    conda clean -afy

#RUN conda run -n llava_yaml pip install flash-attn --no-build-isolation --no-cache-dir
RUN if command -v nvcc >/dev/null 2>&1; then \
    echo "✅ CUDA detected — installing flash-attn..." && \
    conda run -n myenv pip install flash-attn --no-build-isolation --no-cache-dir; \
    else \
    echo "⚠️  Skipping flash-attn install — CUDA not found"; \
    fi

#-------------------------------------------------
# 5) SAMSWISE installation
#-------------------------------------------------

WORKDIR /storage/itaytuviah/

COPY samwise_tools/ /tmp/samwise_tools/

# clone https://github.com/ClaudiaCuttano/SAMWISE.git
RUN cd /tmp && \
    git clone https://github.com/ClaudiaCuttano/SAMWISE.git && \
    cd /tmp/SAMWISE && \
    cp -r /tmp/samwise_tools/* /tmp/SAMWISE/ && \
    rm -rf /tmp/samwise_tools && \
    mkdir -p pretrain && \ 
    cd pretrain && \
    conda run -n myenv gdown --fuzzy https://drive.google.com/file/d/1Molt2up2bP41ekeczXWQU-LWTskKJOV2/view?usp=sharing && \ 
    cd /tmp/SAMWISE && \
    echo '--- pyproject.toml ---' && nl -ba pyproject.toml && \
    conda run -n myenv pip install -e . 

# Set the environment variable to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=
WORKDIR /storage/itaytuviah/Wan2.1
#-------------------------------------------------
# 6) Entry point
#-------------------------------------------------
# Instead of "source activate", which can be tricky in non-interactive shells,
# use 'conda run' to ensure the environment is active when your script runs.
# run python main.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "main.py"]