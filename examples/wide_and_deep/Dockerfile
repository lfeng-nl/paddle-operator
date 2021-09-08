FROM ubuntu:18.04

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list

RUN apt update && \
    apt install -y vim

RUN apt install -y python3 python3-dev python3-pip

RUN python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

RUN python3 -m pip install --no-cache-dir -U six -i https://mirror.baidu.com/pypi/simple

WORKDIR /app

COPY train.py /app
COPY model.py /app
COPY reader.py /app
ADD  data /app/data

ENTRYPOINT ["python3", "train.py"]
