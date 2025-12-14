import sys
import types
import os
from typing_extensions import override

from comfy_api.latest import ComfyExtension

from .checkpoint_loader import LoaderExtension as _CheckpointExtension
from .diffusion_model_loader import DiffusionModelExtension as _DiffusionExtension
from .lora_loader import LoraExtension as _LoraExtension
from .vae_loader import VAEExtension as _VAEExtension

# Expose web assets so ComfyUI loads the tree-selector JS.
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")


class SKLoaderExtension(ComfyExtension):
    def __init__(self):
        self._exts = [
            _CheckpointExtension(),
            _DiffusionExtension(),
            _LoraExtension(),
            _VAEExtension(),
        ]

    @override
    async def get_node_list(self) -> list[type]:
        nodes: list[type] = []
        for ext in self._exts:
            nodes.extend(await ext.get_node_list())
        return nodes


async def comfy_entrypoint() -> SKLoaderExtension:
    return SKLoaderExtension()
