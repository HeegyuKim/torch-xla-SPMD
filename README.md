# torch-xla-SPMD
Pytorch/XLA SPMD Test code in Google TPU 


## Setup
### Pytorch/XLA setup in TPU
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
