FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo vim wget git libsndfile1 espeak-ng sox zip && \
    apt-get clean && \
    pip install -r /tmp/requirements.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu116 && \
    rm /tmp/requirements.txt