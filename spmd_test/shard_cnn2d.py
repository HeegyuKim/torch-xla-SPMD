import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh
import time
from tqdm import tqdm


# Enable XLA SPMD execution mode.
xr.use_spmd()

# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()
mesh_shape = (2, 4, 1)  
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))

# in, out, kernel
model = nn.Conv1d(10, 2, (4, 4)).to(xm.xla_device())
print(model.weight.shape) # 2, 10, 4

model_partition_spec = ('fsdp', 'mp', None, None)
xs.mark_sharding(model.weight, mesh, model_partition_spec)
print(model.weight)


# Training loop
optimizer = optim.AdamW(model.parameters())
model.train()


# data_partition_spec = ('dp', None, 'mp', 'fsdp')
data_partition_spec = ('dp', None, 'fsdp', 'mp')
data = torch.randn(200, 3, 10, 10).to(xm.xla_device())
xs.mark_sharding(data, mesh, data_partition_spec)

start = time.time()
for i in tqdm(range(100)):
    optimizer.zero_grad()

    # batch, feature, seq
    output = model(data)
    loss = output.mean()
    loss.backward()

    optimizer.step()
    xm.mark_step()
    print(loss)

print(time.time() - start)