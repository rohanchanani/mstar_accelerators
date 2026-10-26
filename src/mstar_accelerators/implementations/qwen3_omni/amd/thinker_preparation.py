"""Build the accelerator-owned Thinker from logical checkpoint tensors."""
from copy import deepcopy
import torch
from mstar.model.loader import load_hf_weights
from mstar.model.components.moe import prepare_moe_weights
from .preparation import stacked_parameter_rules, remap_checkpoint_name
from .thinker_model import Qwen3OmniThinkerModel
from .thinker_program import ThinkerProgram
from .moe import select_expert_kernels


def prepare_thinker(binding, resources):
    config = deepcopy(binding.model.config)
    with torch.device("meta"):
        model = Qwen3OmniThinkerModel(config, comm_group=binding.tp_group)
    model.to(dtype=binding.dtype).to_empty(device=binding.device)
    source = binding.model.logical_weights()
    values = source.load(name for name in source.by_name if name.startswith(("thinker.model.", "thinker.lm_head.")))
    loaded = load_hf_weights(model,
        ((name.removeprefix("thinker."), value.to(binding.device))
         for name, value in values.items()),
        stacked_params=stacked_parameter_rules(config.thinker_text.num_experts),
        name_remapper=remap_checkpoint_name)
    missing = set(dict(model.named_parameters())) - loaded
    if missing:
        raise ValueError(f"Uninitialized Thinker parameters: {sorted(missing)}")
    select_expert_kernels(model)
    prepare_moe_weights(model)
    program = ThinkerProgram(model.eval(), config)
    program.bind_resources(resources)
    return program.eval()
