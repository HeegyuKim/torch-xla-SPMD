import os
os.environ["PJRT_DEVICE"] = "TPU"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.use_spmd()

import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh

from transformers import AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel
from peft import LoraConfig, TaskType, get_peft_model
from spmd_util import partition_module

# Enable XLA SPMD execution mode.
# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()
# mesh_shape = (1, 2, num_devices // 2)  # 2x4 on v3-8, 2x2 on v4-8  
mesh_shape = (1, num_devices, 1)  # 2x4 on v3-8, 2x2 on v4-8  
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))

# EleutherAI/pythia-70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b
# EleutherAI/polyglot-ko-12.8b
model_id = "EleutherAI/polyglot-ko-5.8b" 
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config)

# LoRA
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

partition_module(model, mesh)

batch_size = 1
seq_length = 2048

def train():
    optimizer = optim.AdamW(model.parameters())

    # Training loop
    model.train()

    for i in range(10):
        input_ids = torch.randint(0, 1000, (batch_size, seq_length + 1)).to(xm.xla_device())
        xs.mark_sharding(input_ids, mesh, (0, 1))

        output = model(input_ids=input_ids[:, :-1], labels=input_ids[:, 1:])
        loss = output.loss
        loss.backward()

        optimizer.step()
        xm.mark_step()
        optimizer.zero_grad()

        print(f"step {i}", loss.detach().cpu().item())

@torch.no_grad()
def eval():
    model.eval()

    input_ids = torch.randint(0, 1000, (batch_size, seq_length + 1)).to(xm.xla_device())
    xs.mark_sharding(input_ids, mesh, (0, 1))

    for i in range(10):
        output = model(input_ids=input_ids[:, :-1], labels=input_ids[:, 1:])
        loss = output.loss

        print(f"eval loss", loss.cpu().item())

eval()
train()