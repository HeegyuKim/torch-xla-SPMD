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

# Enable XLA SPMD execution mode.
xr.use_spmd()

# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()
mesh_shape = (2, num_devices // 2)  # 2x4 on v3-8, 2x2 on v4-8  
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

t = torch.randn(8, 4).to(xm.xla_device())

# Mesh partitioning, each device holds 1/8-th of the input
partition_spec = (0, 1)
m1_sharded = xs.mark_sharding(t, mesh, partition_spec)
assert isinstance(m1_sharded, XLAShardedTensor) == True
# Note that the sharding annotation is also in-placed updated to t

# Sharding annotate the linear layer weights. SimpleLinear() is a nn.Module.
model = nn.Linear(10, 2).to(xm.xla_device())
xs.mark_sharding(model.weight, mesh, partition_spec)
print(model.weight)
optimizer = optim.AdamW(model.parameters())

# Training loop
model.train()
optimizer.zero_grad()

data = torch.randn(4, 10).to(xm.xla_device())
xs.mark_sharding(data, mesh, partition_spec)
output = model(data)
loss = output.mean()
loss.backward()

optimizer.step()
xm.mark_step()

print(loss)
print(model.weight)