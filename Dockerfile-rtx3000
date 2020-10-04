FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
apt update && \
apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    scikit-learn \
    simpletransformers

RUN python3 -m pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html

RUN mkdir /electra
WORKDIR /electra

CMD ["python3", "classification.py"]