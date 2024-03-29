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

from transformers import LlavaForConditionalGeneration, AutoConfig, AutoProcessor, LlavaConfig, LlavaProcessor, CLIPImageProcessor, AutoTokenizer
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

device = xm.xla_device()

# model_id = "llava-hf/llava-1.5-7b-hf"
# config = AutoConfig.from_pretrained(model_id)
# processor = AutoProcessor.from_pretrained(model_id)
# config.text_config = AutoConfig.from_pretrained(config.text_config._name_or_path)
# config.text_config.num_hidden_layers = 2 # reduce layer size
# config.text_config.hidden_size = 4096

# vocab_size = 32001

vis_config = AutoConfig.from_pretrained("openai/clip-vit-large-patch14").vision_config
text_config = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
text_config.num_hidden_layers = 2 # reduce layer size
vocab_size = text_config.vocab_size
config = LlavaConfig(
    vis_config,
    text_config,
    vocab_size=vocab_size,
    image_token_index=vocab_size - 1
)
print(config)

model = LlavaForConditionalGeneration(config)
print(model)

processor = LlavaProcessor(
    CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14"),
    AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
)

# LoRA
# peft_config = LoraConfig(task_type=TaskType., inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()


url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prefix = "USER: "
text = "\nWhat's the content of the image?\nASSISTANT:"

partition_module(model, mesh, verbose=False)
# xs.mark_sharding(inputs.pixel_values, mesh, ('dp', 'fsdp', None, None))
# xs.mark_sharding(inputs.input_ids, mesh, ('dp', 'fsdp',))
# xs.mark_sharding(inputs.attention_mask, mesh, ('dp', 'fsdp',))

def build_input_embeds(processor, model, prefix, image, texts, labels):
    embeddings = model.get_input_embeddings()
    batch_size = len(texts)

    prefix = processor.tokenizer([prefix] * batch_size, return_tensors="pt")["input_ids"].to(device)
    prefix_embeds = embeddings(prefix)

    texts = processor.tokenizer(texts, return_tensors="pt")["input_ids"].to(device)
    texts_embeds = embeddings(texts)
    
    pixel_values = processor.image_processor(image, return_tensors="pt")["pixel_values"].to(device)

    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = model.config.vision_feature_select_strategy

    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    # for i, hid in enumerate(image_outputs.hidden_states):
    #     print("image_outputs.hidden_states", i, hid.shape)

    selected_image_feature = model.multi_modal_projector(selected_image_feature)

    if vision_feature_select_strategy == "default":
        selected_image_feature = selected_image_feature[:, 1:]
    elif vision_feature_select_strategy == "full":
        selected_image_feature = selected_image_feature

    # print("prefix", prefix_embeds.shape)
    # print("selected_image_feature", selected_image_feature.shape)
    # print("texts", texts_embeds.shape)

    input_embeds = torch.concat([prefix_embeds, selected_image_feature, texts_embeds], dim=1)
    labels = torch.concat([
        torch.full_like(prefix, -100),
        torch.full(selected_image_feature.shape[:2], -100, device=texts.device), 
        texts
        ], dim=1)
    
    return input_embeds, labels

def train():
    optimizer = optim.AdamW(model.parameters())

    # Training loop
    model.train()

    for i in range(100):
        input_embeds, labels = build_input_embeds(
            processor,
            model,
            prefix,
            [image],
            [text],
            [text]
        )
        output = model(inputs_embeds=input_embeds, labels=labels)
        loss = output.loss
        loss.backward()

        optimizer.step()
        xm.mark_step()

        optimizer.zero_grad()

        print(f"step {i}", loss.detach().cpu().item())

# @torch.no_grad()
# def eval():
#     model.eval()

#     for i in tqdm(range(10)):
#         output = model(**inputs)
#         xm.mark_step()

# eval()
train()