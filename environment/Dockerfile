FROM python:3.10.8

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo vim wget git libsndfile1 espeak-ng sox zip && \
    apt-get clean && \
    pip install -r /tmp/requirements.txt --no-cache-dir && rm /tmp/requirements.txt