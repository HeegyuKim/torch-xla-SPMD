# torch-xla-SPMD
- Pytorch/XLA SPMD Test code in Google TPU.
- 
- SPMD는 정식 release에서 지원 안함. nightly 빌드를 써야하는데 docker 쓰는거 추천

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

### pip
에러남... 확인중 일단 도커쓰세요

Guides: 
- https://cloud.google.com/tpu/docs/run-calculation-pytorch?hl=ko#pjrt
- https://github.com/pytorch/xla

```
# 3.10 not working
pip install torch~=2.0.0 https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
pip install torch https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl torch_xla[tpuvm]

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl

# python 3.8
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install torch~=2.0.0 https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl PyYAML numpy

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

export PJRT_DEVICE=TPU
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

```
