
pip install torch https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl torch_xla[tpuvm]

echo 'export PJRT_DEVICE=TPU' >> ~/.bashrc
export PJRT_DEVICE=TPU
