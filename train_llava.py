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

from transformers import LlavaForConditionalGeneration, AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model
from spmd_util import partition_module
from PIL import Image
import requests
from tqdm import tqdm


# Enable XLA SPMD execution mode.
# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()
# mesh_shape = (1, 2, num_devices // 2)  # 2x4 on v3-8, 2x2 on v4-8  
mesh_shape = (1, num_devices, 1)  # 2x4 on v3-8, 2x2 on v4-8  
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))

# EleutherAI/pythia-70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b
# EleutherAI/polyglot-ko-12.8b
model_id = "llava-hf/llava-1.5-7b-hf"
config = AutoConfig.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
config.text_config.num_hidden_layers = 2 # reduce layer size
print(config)

model = LlavaForConditionalGeneration(config)
print(model)

# LoRA
# peft_config = LoraConfig(task_type=TaskType., inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()


url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
inputs = processor(images=image, text=text, return_tensors="pt")

partition_module(model, mesh, verbose=True)
inputs = inputs.to(xm.xla_device())
xs.mark_sharding(inputs.pixel_values, mesh, ('dp', 'fsdp', None, None))
xs.mark_sharding(inputs.input_ids, mesh, ('dp', 'fsdp',))
xs.mark_sharding(inputs.attention_mask, mesh, ('dp', 'fsdp',))

print(inputs)
inputs.pop("pixel_values")

def train():
    optimizer = optim.AdamW(model.parameters())

    # Training loop
    model.train()

    for i in range(10):

        output = model(**inputs, labels=inputs["input_ids"])
        loss = output.loss
        loss.backward()

        optimizer.step()
        xm.mark_step()

        optimizer.zero_grad()

        print(f"step {i}", loss.detach().cpu().item())

@torch.no_grad()
def eval():
    model.eval()

    for i in tqdm(range(10)):
        output = model(**inputs)
        print(output)
        xm.mark_step()

eval()
# train()