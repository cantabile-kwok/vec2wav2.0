FROM python:3.10.8

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo vim wget git libsndfile1 espeak-ng sox zip && \
    apt-get clean && \
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install -r /tmp/requirements.txt --no-cache-dir && rm /tmp/requirements.txt