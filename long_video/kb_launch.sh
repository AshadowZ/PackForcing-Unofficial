script=$1
nodes=$2
name=$3

mode=$4
model=$5
port=$6

folder="kb_configs"

# 检查文件夹是否存在
if [ ! -d "$folder" ]; then
    # 如果文件夹不存在，则创建它
    mkdir "$folder"
fi

current_date=$(date +"%Y-%m-%d-%H-%M-%S-%6N")



if [ -n "$script" ]; then
    dest="$folder/${name}_${current_date}.sh"
    if [ -e "$dest" ]; then
        echo "$dest 已存在，退出脚本"
        exit 1
    fi
    echo "复制脚本 $script 至 $dest"
    cp $script $dest
    script=$dest
    script=$(readlink -f "${script}")
    echo "执行脚本： $script"
else
    echo "没有输入执行脚本，退出"
    exit 1
fi


if [ -n "$nodes" ]; then
    echo "节点数： $nodes"
else
    echo "没有输入节点数，退出"
    exit 1
fi


if [ -n "$name" ]; then
    echo "任务名： $name"
else
    echo "没有输入任务名，退出"
    exit 1
fi


yaml_single="apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: ${name}  # 需要修改
  namespace: default
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        metadata: null
        spec:
          tolerations:
             - key: audio
               operator: Exists
               effect: NoSchedule
          containers:
          - command:  # 需要修改
            - /bin/sh
            - -c
            - . /opt/conda/etc/profile.d/conda.sh
              && source /mnt/vepfs/base2/zhuhongzhou/anaconda3/bin/activate
              && conda activate causal_forcing
              && bash ${script} \$HOSTNAME "$mode" "$model" "$port"
            env:
              - name: NCCL_DEBUG
                value: INFO
              - name: NCCL_DEBUG_FILE
                value: /mnt/vepfs/nieshen/nccl-logs/nccl.log.%h.%p
              - name: NCCL_NET_PLUGIN
                value: none
              - name: NCCL_IB_TIMEOUT
                value: \"22\"
              - name: MASTER
                value: \"1\"
            image: registry.baidubce.com/cce-ai-native/pytorch:23.03-py3-cn
            imagePullPolicy: IfNotPresent
            name: pytorch
            resources:
              limits:
                cpu: 100
                memory: 800Gi
                baidu.com/a800_80g_cgpu: 8
                rdma/hca: 1
            securityContext:
              capabilities:
                add:
                - IPC_LOCK
            volumeMounts:
              - mountPath: /dev/shm
                name: cache-volume
              - mountPath: /mnt/vepfs
                name: model-pfs
          dnsPolicy: ClusterFirstWithHostNet
          hostNetwork: true
          schedulerName: volcano
          volumes:
          - emptyDir:
              medium: Memory
            name: cache-volume
          - name: model-pfs
            hostPath:
              path: /mnt/vepfs
              type: Directory
  runPolicy:
    schedulingPolicy:
      queue: embody
"

# yaml_single="apiVersion: kubeflow.org/v1
# kind: PyTorchJob
# metadata:
#   annotations:
#     aijob.cce.baidubce.com/fault-tolerance-enabled: 'true'
#     aijob.cce.baidubce.com/fault-tolerance-terminating-force-delete-minutes: '3'
#     aijob.cce.baidubce.com/barrier-timeout-seconds: '60'
#   namespace: default
#   name: ${name}
# spec:
#   pytorchReplicaSpecs:
#     Master:
#       replicas: 1
#       restartPolicy: Never
#       template:
#         metadata: null
#         spec:
#           tolerations:
#           - key: debug
#             operator: Exists
#             effect: NoSchedule
#           imagePullSecrets:
#           - name: registry-secrets
#           containers:
#           - command:
#             - /bin/sh
#             - -c
#             - 'bash scripts/test_a800.sh '
#             env:
#             - name: NCCL_NET_PLUGIN
#               value: none
#             - name: NCCL_IB_TIMEOUT
#               value: '22'
#             - name: NCCL_DEBUG
#               value: ERROR
#             - name: MASTER
#               value: '1'
#             - name: TORCH_HOME
#               value: /mnt/vepfs/base/chenyimin/.cache/torch
#             workingDir: /mnt/vepfs/base/chenyimin/codebases/ss-falcon
#             image: ccr-2jcv050f-vpc.cnc.bj.baidubce.com/train/ss-vidu:v0.0.1
#             imagePullPolicy: IfNotPresent
#             name: pytorch
#             resources:
#               limits:
#                 cpu: 100
#                 memory: 800Gi
#                 baidu.com/a800_80g_cgpu: 8
#                 rdma/hca: 1
#             securityContext:
#               capabilities:
#                 add:
#                 - IPC_LOCK
#             volumeMounts:
#             - mountPath: /dev/shm
#               name: cache-volume
#             - mountPath: /mnt/vepfs
#               name: model-pfs
#           dnsPolicy: ClusterFirstWithHostNet
#           hostNetwork: true
#           schedulerName: volcano
#           volumes:
#           - emptyDir:
#               medium: Memory
#             name: cache-volume
#           - name: model-pfs
#             hostPath:
#               path: /mnt/vepfs
#               type: Directory
#   runPolicy:
#     schedulingPolicy:
#       queue: embody
# "

yaml_multi="apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: ${name}  # 需要修改
  namespace: default
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        metadata: null
        spec:
          tolerations:
             - key: audio
               operator: Exists
               effect: NoSchedule
          containers:
          - command:  # 需要修改
            - /bin/sh
            - -c
            - . /opt/conda/etc/profile.d/conda.sh
              && source /mnt/vepfs/base2/zhuhongzhou/anaconda3/bin/activate
              && conda activate causal_forcing
              && bash ${script} \$HOSTNAME "$mode" "$model" "$port"
            env:
              - name: NCCL_DEBUG
                value: INFO
              - name: NCCL_DEBUG_FILE
                value: /mnt/vepfs/nieshen/nccl-logs/nccl.log.%h.%p
              - name: NCCL_NET_PLUGIN
                value: none
              - name: NCCL_IB_TIMEOUT
                value: \"22\"
              - name: MASTER
                value: \"1\"
            image: registry.baidubce.com/cce-ai-native/pytorch:23.03-py3-cn
            imagePullPolicy: IfNotPresent
            name: pytorch
            resources:
              limits:
                cpu: 100
                memory: 800Gi
                baidu.com/a800_80g_cgpu: 8
                rdma/hca: 1
            securityContext:
              capabilities:
                add:
                - IPC_LOCK
            volumeMounts:
              - mountPath: /dev/shm
                name: cache-volume
              - mountPath: /mnt/vepfs
                name: model-pfs
          dnsPolicy: ClusterFirstWithHostNet
          hostNetwork: true
          schedulerName: volcano
          volumes:
          - emptyDir:
              medium: Memory
            name: cache-volume
          - name: model-pfs
            hostPath:
              path: /mnt/vepfs
              type: Directory
    Worker:
      replicas: $((${nodes} - 1))  # worker为3代表32张卡
      restartPolicy: Never
      template:
        spec:
          containers:
          - command:  # 需要修改
            - /bin/sh
            - -c
            - . /opt/conda/etc/profile.d/conda.sh
              && source /mnt/vepfs/base2/zhuhongzhou/anaconda3/bin/activate
              && conda activate causal_forcing
              && bash ${script} \$MASTER_ADDR
            env:
              - name: NCCL_DEBUG
                value: INFO
              - name: NCCL_DEBUG_FILE
                value: /mnt/vepfs/nieshen/nccl-logs/nccl.log.%h.%p
              - name: NCCL_NET_PLUGIN
                value: none
              - name: NCCL_IB_TIMEOUT
                value: \"22\"
            image: registry.baidubce.com/cce-ai-native/pytorch:23.03-py3-cn
            imagePullPolicy: IfNotPresent
            name: pytorch
            resources:
              limits:
                cpu: 100
                memory: 800Gi
                baidu.com/a800_80g_cgpu: 8
                rdma/hca: 1
            securityContext:
              capabilities:
                add:
                - IPC_LOCK
            volumeMounts:
              - mountPath: /dev/shm
                name: cache-volume
              - mountPath: /mnt/vepfs
                name: model-pfs
          dnsPolicy: ClusterFirstWithHostNet
          hostNetwork: true
          schedulerName: volcano
          volumes:
          - emptyDir:
              medium: Memory
            name: cache-volume
          - name: model-pfs
            hostPath:
              path: /mnt/vepfs
              type: Directory
  runPolicy:
    schedulingPolicy:
      queue: embody
"


if [ "$nodes" -eq 1 ]; then
    yaml=${yaml_single}
else
    yaml=${yaml_multi}
fi


out_yaml="$folder/${name}_${current_date}.yaml"

echo "$yaml" > $out_yaml
echo "保存yaml配置到 ${out_yaml}"
kubectl apply -f $out_yaml
