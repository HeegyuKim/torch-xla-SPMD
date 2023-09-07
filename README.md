# torch-xla-SPMD
Pytorch/XLA SPMD Test code in Google TPU 

- SPMD는 정식 release에서 지원 안함. nightly 빌드를 써야하는데 docker 쓰는거 추천

## Setup

### Use docker
docker without sudo
```
sudo groupadd docker
sudo usermod -aG docker ${USER}
sudo service docker restart
```

run xla docker
```
sudo docker run -it --name torch \
    -d --privileged \
    -p 7860:7860 \
    -v `pwd`:/workspace \
    us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_20230901 \
    /bin/bash

sudo docker run -ti --rm --name your-container-name --privileged -v `pwd`:/workspace gcr.io/tpu-pytorch/xla:r2.0_3.8_tpuvm bash
```

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
