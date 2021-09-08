from paddle.distributed.fleet.utils.ps_util import DistributedInfer
import paddle.distributed.fleet as fleet
import numpy as np
from model import WideDeepModel

import os
import sys

import paddle
paddle.enable_static()

def distributed_training(exe, model, train_data_path="./data", batch_size=4, epoch_num=10):

    # if you want to use InMemoryDataset, please invoke load_into_memory/release_memory at train_from_dataset front and back.
    #dataset = paddle.distributed.InMemoryDataset()
    #dataset.load_into_memory()
    # train_from_dataset ...

    dataset = paddle.distributed.QueueDataset()
    thread_num = 2
    dataset.init(use_var=model.inputs, pipe_command="python reader.py", batch_size=batch_size, thread_num=thread_num)
    #dataset.load_into_memory()

    train_files_list = [os.path.join(train_data_path, x)
                          for x in os.listdir(train_data_path)]


    for epoch_id in range(epoch_num):
        dataset.set_filelist(train_files_list)
        exe.train_from_dataset(paddle.static.default_main_program(),
                               dataset,
                               paddle.static.global_scope(),
                               #debug=True,
                               #thread=4,
                               fetch_list=[model.cost, model._metrics["auc"]["result"]],
                               fetch_info=["loss", "auc"],
                               print_period=1)
        #dataset.release_memory()

def distributed_infer(exe, model, train_data_path="./data", batch_size=10):

    dataset = paddle.distributed.QueueDataset()
    thread_num = 2
    dataset.init(use_var=model.inputs, pipe_command="python reader.py", batch_size=batch_size, thread_num=thread_num)

    train_files_list = [os.path.join(train_data_path, x)
                          for x in os.listdir(train_data_path)]

    dataset.set_filelist(train_files_list)
    exe.infer_from_dataset(paddle.static.default_main_program(),
                           dataset,
                           paddle.static.global_scope(),
                           debug=False,
                           fetch_list=[model.cost, model._metrics["auc"]["result"]],
                           fetch_info=["loss", "auc"],
                           print_period=1)

def clear_metric_state(model, place):
    for metric_name in model._metrics:
        for _, state_var_tuple in model._metrics[metric_name]["state"].items():
            var = paddle.static.global_scope().find_var(
                state_var_tuple[0].name)
            if var is None:
                continue
            var = var.get_tensor()
            data_zeros = np.zeros(var._get_dims()).astype(state_var_tuple[1])
            var.set(data_zeros, place)


fleet.init(is_collective=False)

model = WideDeepModel()
model.net(is_train=True)

strategy = fleet.DistributedStrategy()
strategy.a_sync = True

optimizer = paddle.optimizer.SGD(learning_rate=0.0001)

optimizer = fleet.distributed_optimizer(optimizer, strategy)

optimizer.minimize(model.cost)


if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
    print("server stoped")

if fleet.is_worker():
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    exe.run(paddle.static.default_startup_program())

    fleet.init_worker()

    print("begin trainning")
    distributed_training(exe, model)

    clear_metric_state(model, place)

    print("begin infering")
    distributed_infer(exe, model)

    fleet.stop_worker()

    fleet.barrier_worker()
    if int(fleet.worker_index()) == 0:
        fleet.fleet._runtime_handle._worker.stop_server()

    print("worker stoped")
