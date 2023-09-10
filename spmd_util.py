import torch
import torch.nn as nn
import re
import torch_xla.experimental.xla_sharding as xs
import torch_xla.core.xla_model as xm

GPTNEOX_RULES = (
    # embeddings
    ("gpt_neox\\.embed_in", ("mp", "fsdp")),
    # atention
    ("attention\\.query_key_value$", ("fsdp", "mp")),
    ("attention\\.dense$", ("mp", "fsdp")),
    # mlp
    ("mlp\\.dense_h_to_4h$", ("fsdp", "mp")),
    ("mlp\\.dense_4h_to_h$", ("mp", "fsdp")),
    # output
    ("embed_out", ("fsdp", "mp")),
)
strkey2id = {
    "dp": 0,
    "fsdp": 1,
    "mp": 2
}

def partition_module(model, mesh, partition_specs=GPTNEOX_RULES, device=xm.xla_device()):
    rule = [(k, tuple([strkey2id[x] for x in v])) for k, v in partition_specs]
        
    # print(rule)

    for name, module in model.named_modules():
        module.to(device)
        # print(name, module.__class__.__name__)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            for rule_pattern, spec in rule:
                if re.findall(rule_pattern, name):
                    print("match", rule_pattern, name)
                    
                    xs.mark_sharding(module.weight, mesh, spec)
                    break
        