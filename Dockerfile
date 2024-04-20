FROM --platform=linux/x86_64 thr3a/cuda12.1-torch:latest

# RUN apt-get update \
#   && apt-get install --no-install-recommends -y git \
#   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/CartoonSegmentation/CartoonSegmentation.git ./

RUN pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
RUN mim install mmdet
RUN apt-get update \
  && apt-get install --no-install-recommends -y python3.11-dev \
  && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
