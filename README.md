# Paddle Operator

## Overview

Paddle Operator makes it easy to run [paddle](https://www.paddlepaddle.org.cn/)
distributed training job on kubernetes by providing PaddleJob custom resource etc.

## Quick Start
### Prerequisites

* Kubernetes >= 1.8
* kubectl

### Installation

With kubernetes ready, you can install paddle operator with configuration in *deploy* folder 
(use *deploy/v1* for kubernetes v1.16+ or *deploy/v1beta1* for kubernetes 1.15-).

Create PaddleJob crd,
```shell
kubectl apply -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/main/deploy/v1/crd.yaml
```

A succeed creation leads to result as follows,
```shell
kubectl get crd
NAME                                    CREATED AT
paddlejobs.batch.paddlepaddle.org       2021-02-08T07:43:24Z
```

Then deploy controller,

```shell
kubectl apply -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/main/deploy/v1/operator.yaml
```

the ready state of controller would be as follow,
```shell
kubectl -n paddle-system get pods
NAME                                         READY   STATUS    RESTARTS   AGE
paddle-controller-manager-698dd7b855-n65jr   1/1     Running   0          1m
```

By default, paddle controller runs in namespace *paddle-system* and only controls jobs in that namespace.
To run controller in a different namespace or controll jobs in other namespaces, you can edit `charts/paddle-operator/values.yaml` and install the helm chart.
You can also edit kustomization files or edit `deploy/v1/operator.yaml` directly for that purpose.

### Run demo paddlejob

Deploy your first paddlejob demo with
```shell
kubectl -n paddle-system apply -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/main/deploy/examples/wide_and_deep.yaml
```

Check pods status
```shell
kubectl -n paddle-system get pods
```

Check paddle job status
```shell
kubectl -n paddle-system get pdj
```

### Work with Volcano

Enable volcano before installation, add the following args in *deploy/v1/operator.yaml*
```
containers:
- args:
  - --leader-elect
  - --namespace=paddle-system  # watch this ns only
  - --scheduling=volcano       # enable volcano
  command:
  - /manager
```

then, job as in *deploy/examples/wide_and_deep_volcano.yaml* can be handled correctly.

### Elastic Trainning

Elastic feature depend on etcd present, which should be set for controller as args,
```
  --etcd-server=paddle-elastic-etcd.paddle-system.svc.cluster.local:2379      # enable elastic
```

then, job as in *deploy/elastic/resnet.yaml* can be handled correctly.

### Deploy args
Change the following args in *deploy/v1/operator.yaml* before deployment,
```
- args:
  - --leader-elect             # enable leader election
  - --namespace=paddle-system  # watch this ns only, set to "" for all namespace
  - --scheduling=volcano       # enable volcano
  - --initImage=""             # init container image, default to busybox, "" to disable
  command:
  - /manager
```

### Uninstall
Simply
```shell
kubectl delete -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/main/deploy/v1/crd.yaml -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/main/deploy/v1/operator.yaml
```
## Advanced usage

More configuration can be found in Makefile, clone this repo and enjoy it.
If you have any questions or concerns about the usage, please do not hesitate to contact us.

## More Information

Please refer to the
[中文文档](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/paddle_on_k8s.html) 
for more information about paddle configuration.
