# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class DeepFMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes):
        super(DeepFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes

        self.fm = FM(sparse_feature_number, sparse_feature_dim,
                     dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim,
                       dense_feature_dim, dense_feature_dim + sparse_num_field,
                       layer_sizes)
        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order, feat_embeddings = self.fm(sparse_inputs,
                                                                 dense_inputs)
        y_dnn = self.dnn(feat_embeddings)

        predict = F.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict


class FM(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1
        # sparse coding
        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            padding_idx=0,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        # dense coding
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))

        self.dense_w = paddle.create_parameter(
            shape=[1, self.dense_feature_dim, self.dense_emb_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)
        dense_emb_one = paddle.unsqueeze(dense_emb_one, axis=2)

        y_first_order = paddle.sum(sparse_emb_one, 1) + paddle.sum(
            dense_emb_one, 1)

        # -------------------- second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)
        # sum_square part
        summed_features_emb = paddle.sum(feat_embeddings,
                                         1)  # None * embedding_size
        summed_features_emb_square = paddle.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = paddle.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = paddle.sum(squared_features_emb,
                                              1)  # None * embedding_size

        y_second_order = 0.5 * paddle.sum(
            summed_features_emb_square - squared_sum_features_emb,
            1,
            keepdim=True)  # None * 1

        return y_first_order, y_second_order, feat_embeddings


class DNN(paddle.nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(DNN, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        sizes = [sparse_feature_dim * num_field] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)

    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
        return y_dnn

class DeepFMModel:
    def __init__(self, sparse_feature_number=1000001, sparse_inputs_slots=27, sparse_feature_dim=8, dense_input_dim=13, fc_sizes=[400, 400, 400]):
        self.sparse_feature_number = sparse_feature_number
        self.sparse_inputs_slots = sparse_inputs_slots
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_input_dim = dense_input_dim
        self.fc_sizes = fc_sizes

        self._metrics = {}

    def acc_metrics(self, pred, label):
        correct_cnt = paddle.static.create_global_var(
            name="right_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_accuracy = paddle.static.accuracy(input=pred, label=label)
        batch_correct = batch_cnt * batch_accuracy

        paddle.assign(correct_cnt + batch_correct, correct_cnt)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        accuracy = correct_cnt / total_cnt

        self._metrics["acc"] = {}
        self._metrics["acc"]["result"] = accuracy
        self._metrics["acc"]["state"] = {
            "total": (total_cnt, "float32"), "correct": (correct_cnt, "float32")}

    def auc_metrics(self, pred, label):
        auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] = paddle.static.auc(input=pred,
                                                                                                 label=label,
                                                                                                 num_thresholds=2**12,
                                                                                                 slide_steps=20)

        self._metrics["auc"] = {}
        self._metrics["auc"]["result"] = auc
        self._metrics["auc"]["state"] = {"stat_pos": (
            stat_pos, "int64"), "stat_neg": (stat_neg, "int64")}

    def mae_metrics(self, pred, label):
        abserr = paddle.static.create_global_var(
            name="abserr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_abserr = paddle.nn.functional.l1_loss(
            pred, label, reduction='sum')

        paddle.assign(abserr + batch_abserr, abserr)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        mae = abserr / total_cnt

        self._metrics["mae"] = {}
        self._metrics["mae"]["result"] = mae
        self._metrics["mae"]["state"] = {
            "total": (total_cnt, "float32"), "abserr": (abserr, "float32")}

    def mse_metrics(self, pred, label):
        sqrerr = paddle.static.create_global_var(
            name="sqrerr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_sqrerr = paddle.nn.functional.mse_loss(
            pred, label, reduction='sum')

        paddle.assign(sqrerr + batch_sqrerr, sqrerr)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        mse = sqrerr / total_cnt
        rmse = paddle.sqrt(mse)

        self._metrics["mse"] = {}
        self._metrics["mse"]["result"] = mse
        self._metrics["mse"]["state"] = {
            "total": (total_cnt, "float32"), "sqrerr": (sqrerr, "float32")}

        self._metrics["rmse"] = {}
        self._metrics["rmse"]["result"] = rmse
        self._metrics["rmse"]["state"] = {
            "total": (total_cnt, "float32"), "sqrerr": (sqrerr, "float32")}

    def net(self, is_train=True):

        dense_input = paddle.static.data(name="dense_input", shape=[
                                         None, self.dense_input_dim], dtype="float32")

        sparse_inputs = [
            paddle.static.data(name="C" + str(i),
                               shape=[None, 1],
                               lod_level=1,
                               dtype="int64") for i in range(1, self.sparse_inputs_slots)
        ]

        label_input = paddle.static.data(
            name="click", shape=[None, 1], dtype="int64")

        self.inputs = [dense_input] + sparse_inputs + [label_input]

        sparse_number = self.sparse_inputs_slots - 1
        assert sparse_number == len(sparse_inputs)

        deepfm_model = DeepFMLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.dense_input_dim, sparse_number, self.fc_sizes)

        pred = deepfm_model.forward(sparse_inputs, dense_input)

        #pred = F.sigmoid(prediction)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc_var, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                        label=label_input,
                                                        slide_steps=0)

        self.inference_target_var = auc
        '''
        if not is_train:
            fetch_dict = {'auc': auc}
            return fetch_dict
        '''

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self.cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'auc': auc}
        return fetch_dict

