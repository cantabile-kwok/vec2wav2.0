FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y sudo vim wget git libsndfile1 espeak-ng

RUN apt-get install -y sox zip


WORKDIR /app/
COPY ./ /app/
RUN pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

