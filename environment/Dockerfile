FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo build-essential vim wget git libsndfile1 espeak-ng sox zip && \
    apt-get clean && \
    pip install -r /tmp/requirements.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu116 && \
    rm /tmp/requirements.txt