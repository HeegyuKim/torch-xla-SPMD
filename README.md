# torch-xla-SPMD
- Pytorch/XLA SPMD Test code in Google TPU.
- For more details, see [SPMD user guide](https://pytorch.org/xla/release/2.1/index.html#pytorch-xla-spmd-user-guide)

## Installation
```
pip install torch~=2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install -r requirements.txt

# If you use conda, set your lib/ 
export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib

# for transformers library in TPU
export USE_TORCH=True
```

## SPMD usage
see [train_with_spmd.py](./train_with_spmd.py)

```python
# 1. define mesh (use fsdp in in this example)
num_devices = xr.global_runtime_device_count()
mesh_shape = (1, num_devices, 1)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))

...

# 2. partition module with mesh
partition_module(model, mesh) 

...

# 3. partition input with mesh and forward
input_ids = torch.randint(0, 1000, (batch_size, seq_length + 1)).to(xm.xla_device())
xs.mark_sharding(input_ids, mesh, (0, 1))

output = model(input_ids=input_ids[:, :-1], labels=input_ids[:, 1:])

```

### implemented sharding rules
You can see the implemented sharding rules in [spmd_util.py](./spmd_util.py)

| Model | Implemented |
| --- | --- | 
| GPT-NeoX | ✅ |
| T5 | ✅ |
| Llama | ✅ |

## SPMD OOM Check (20230901 nightly)
- code: [spmd_gpt.py](spmd_gpt.py)
- Architecture: GPT NeoX
- batch_size: 1
- Optimizer: AdamW
- Mesh: (1, 8, 1). Mesh shape이 바뀌면 OOM이 생기기도 합니다.

### TPU v2-8
| # params | seq_length | Inference  | Trainable / LoRA | 
| --- | --- | --- | --- |
| 1.4B | 2048 | ✅ | ✅ / ✅ |
| 2.8B | 2048 | ✅ | 🤯 / ✅ |
| 3.8B | 2048 | ✅ | 🤯 / ✅ |
| 5.8B | 2048 | ✅(4) | 🤯 / ✅ |
| 6.9B | 2048 | ✅(2) | 🤯 / ✅(2) |
| 12.8B | 2048 | ✅ | 🤯 / ✅ |

() 괄호는 해당 크기 배치로 했을 때 OOM 발생하지 않았음을 표시. 표시 없을 경우 1이고 더 큰 배치에서 안되는 것은 아님. 실험을 안한 것.

### TPU v3-8
| # params | seq_length | Inference  | Trainable / LoRA | 
| --- | --- | --- | --- |
| 1.4B | 2048 | ✅ | ✅ / ✅ |
| 2.8B | 2048 | ✅ | ✅ / ✅ |
| 3.8B | 2048 | ✅ | ✅ / ✅ |
| 5.8B | 2048 | ✅ | 🤯 / ✅ |
| 6.9B | 2048 | ✅(4) | 🤯 / ✅(4) |
| 12.8B | 2048 | ✅(2) | 🤯 / ✅(1) |


## Setup

### Use docker
```
sudo docker run -it --name torch \
    -d --privileged \
    -p 7860:7860 \
    -v `pwd`:/workspace \
    us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_20230901 \
    /bin/bash
```
