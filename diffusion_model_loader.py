from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

import folder_paths

from .tree_utils import attach_tree_metadata, build_tree, list_files, resolve_selected_path


def build_file_input(input_id: str, folder_type: str, tooltip: str | None = None) -> io.Combo.Input:
    options: list[str] = []
    for rel_path in list_files(folder_type, ""):
        options.append(rel_path)
    if not options:
        options = ["<none>"]
    combo = io.Combo.Input(input_id, options=sorted(set(options)), tooltip=tooltip or "Select file")
    tree = build_tree(folder_type, input_id)
    return attach_tree_metadata(combo, tree, tooltip=tooltip or "Select file")


class UNETLoader(io.ComfyNode):
    CATEGORY = "SK Loader/Advanced"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_UNETLoader",
            display_name="[SK] Diffusion Model",
            category="SK Loader/Advanced",
            inputs=[
                build_file_input("unet", "diffusion_models"),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, unet: dict | str, weight_dtype: str) -> io.NodeOutput:
        import torch
        import comfy.sd

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = resolve_selected_path("diffusion_models", unet, "unet", "unet")
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return io.NodeOutput(model)


class DiffusionModelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UNETLoader,
        ]


async def comfy_entrypoint() -> DiffusionModelExtension:
    return DiffusionModelExtension()
