# ------------------------
# 基礎 image
# ------------------------
FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

# ------------------------
# 基本設定
# ------------------------
WORKDIR /root/
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG PYTHON_VERSION=3.9
ENV PATH=/opt/conda/bin:$PATH

# ------------------------
# 安裝系統依賴（OpenCV, build, wget 等）
# ------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        wget \
        bzip2 \
        ca-certificates \
        vim \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        procps \
        openssh-client \
        subversion \
        mercurial \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------
# 安裝 Miniconda 3.9
# ------------------------
ARG MINICONDA_VERSION=py39_4.12.0
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh -q \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && /opt/conda/bin/conda clean -afy

# ------------------------
# 建立 conda env
# ------------------------
RUN conda create -n exp python=${PYTHON_VERSION} -y \
    && conda init bash \
    && echo "conda activate exp" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=exp
ENV PATH=/opt/conda/envs/exp/bin:$PATH

# ------------------------
# 安裝 Python 套件
# ------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ------------------------
# EntryPoint，啟動自動 activate env
# ------------------------
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate exp\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
