FROM paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7

ADD . /app

WORKDIR /app

RUN pip install scipy opencv-python==4.2.0.32 -i https://mirror.baidu.com/pypi/simple

CMD ["python","-m","paddle.distributed.launch","train_fleet_dygraph.py"]
