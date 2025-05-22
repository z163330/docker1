# 使用官方Ubuntu镜像
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /cluster_app

# 更换源为阿里云
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update

# 安装基本依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 下载并安装Miniconda
ENV CONDA_DIR /opt/conda
RUN rm -rf $CONDA_DIR && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# 将conda加入PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# 创建虚拟环境
RUN conda create -n cluster python=3.11 -y

# 安装依赖
COPY requirements.txt .
RUN conda run -n cluster pip install --trusted-host pypi.tuna.tsinghua.edu.cn -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制项目文件
COPY cluster_topic_keypoints cluster_topic_keypoints

# 设置容器启动命令
CMD ["conda", "run", "-n", "cluster", "python", "cluster_topic_keypoints/cluster_topic_lishi.py"]

# 暴露端口
EXPOSE 5688
