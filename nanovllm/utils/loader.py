import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weight_prefix = getattr(model, "weight_prefix", "")
    visual_prefix = getattr(model, "visual_prefix", "")
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Handle visual encoder weights (e.g. "model.visual." -> "visual.")
                if visual_prefix and weight_name.startswith(visual_prefix):
                    mapped_name = weight_name[len(visual_prefix) - len("visual."):]
                    # visual weights map to: visual.* on the model
                    mapped_name = weight_name.replace(visual_prefix, "visual.")
                    try:
                        param = model.get_parameter(mapped_name)
                    except (AttributeError, KeyError):
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    continue

                # Strip model-specific prefix (e.g. "model.language_model." -> "model.")
                mapped_name = weight_name
                if weight_prefix and mapped_name.startswith(weight_prefix):
                    mapped_name = "model." + mapped_name[len(weight_prefix):]

                # Check packed modules
                for k in packed_modules_mapping:
                    if k in mapped_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = mapped_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except (AttributeError, KeyError):
                            break
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(mapped_name)
                    except (AttributeError, KeyError):
                        continue  # Skip weights not in model (mtp, etc.)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
