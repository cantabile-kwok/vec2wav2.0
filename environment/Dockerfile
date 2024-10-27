FROM ubuntu:18.04

COPY ./requirements.txt /tmp/requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo vim wget git libsndfile1 espeak-ng sox zip python3.10 && \
    apt-get clean && \
    pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt && \
    rm -rf .cache/pip