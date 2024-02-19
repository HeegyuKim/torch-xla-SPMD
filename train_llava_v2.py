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

class XLALlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1).int() >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            print(image_to_overwrite)
            print(image_to_overwrite.sum())
            print(image_features.shape[:-1].numel())
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids
    

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

processor = LlavaProcessor(
    CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14"),
    AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")
)
processor.tokenizer.add_special_tokens({
    "additional_special_tokens": ["<image>"]
})

image_token_id = 32000

vis_config = AutoConfig.from_pretrained("openai/clip-vit-large-patch14").vision_config
text_config = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
text_config.num_hidden_layers = 2 # reduce layer size
vocab_size = text_config.vocab_size
config = LlavaConfig(
    vis_config,
    text_config,
    vocab_size=vocab_size,
    image_token_index=image_token_id
)
print(config)

model = XLALlavaForConditionalGeneration(config)
print(model)

# LoRA
# peft_config = LoraConfig(task_type=TaskType., inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()


url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# partition_module(model, mesh, verbose=False)
model = model.to(device)

def train():
    optimizer = optim.AdamW(model.parameters())

    # Training loop
    model.train()
    text = "User: <image>\nWhat's the content of the image?\nASSISTANT:"
    inputs = processor(images=[image], text=[text]).to(device)
    print(inputs["input_ids"])

    for i in range(100):
        output = model(**inputs, labels=inputs["input_ids"])
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