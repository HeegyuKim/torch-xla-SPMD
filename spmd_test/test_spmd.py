import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharding import Mesh

# Enable XLA SPMD execution mode.
xr.use_spmd()

# Assuming you are running on a TPU host that has 8 devices attached
num_devices = xr.global_runtime_device_count()
# mesh shape will be (4,2) in this example
mesh_shape = (num_devices // 2, 2)
device_ids = np.array(range(num_devices))
# axis_names 'x' nad 'y' are optional
mesh = Mesh(device_ids, mesh_shape, ('model', 'data'))

print(mesh.get_logical_mesh())
print(mesh.shape())


partition_spec = ('model', 'data')

input_tensor = torch.arange(16).unsqueeze(0).repeat(8, 1).to(xm.xla_device())
print(input_tensor)
xs.mark_sharding(input_tensor, mesh, partition_spec)
print(input_tensor)