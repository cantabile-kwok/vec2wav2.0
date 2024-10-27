FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY ./requirements.txt /tmp/requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo vim wget git libsndfile1 espeak-ng sox zip && \
    apt-get clean && \
    pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt