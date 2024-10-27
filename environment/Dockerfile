FROM ubuntu:18.04

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo vim wget git libsndfile1 espeak-ng sox zip python3.10 python3-pip && \
    apt-get clean && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt