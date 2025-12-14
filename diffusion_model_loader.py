from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, _io

import folder_paths

from .tree_utils import attach_tree_metadata, build_tree, list_dirs, list_files, resolve_selected_path, sanitize_rel_dir


def build_folder_file_input(folder_id: str, file_id: str, folder_type: str, tooltip: str | None = None) -> _io.DynamicCombo.Input:
    options: list[_io.DynamicCombo.Option] = []
    for rel_dir in list_dirs(folder_type):
        file_options = list_files(folder_type, rel_dir)
        if not file_options:
            continue
        child_id = f"{file_id}__{sanitize_rel_dir(rel_dir)}"
        inputs = [io.Combo.Input(child_id, options=file_options, tooltip=tooltip)]
        options.append(_io.DynamicCombo.Option(rel_dir if rel_dir else "root", inputs))
    if not options:
        child_id = f"{file_id}__none"
        options.append(_io.DynamicCombo.Option("No files found", [io.Combo.Input(child_id, options=["<none>"], tooltip="No files found")]))
    combo = _io.DynamicCombo.Input(folder_id, options=options, tooltip="Select folder")
    tree = build_tree(folder_type, file_id)
    return attach_tree_metadata(combo, tree, tooltip="Select folder")


class UNETLoader(io.ComfyNode):
    CATEGORY = "SK Loader/Advanced"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_UNETLoader",
            display_name="[SK] Diffusion Model",
            category="SK Loader/Advanced",
            inputs=[
                build_folder_file_input("unet_folder", "unet_name", "diffusion_models"),
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
    def execute(cls, unet_folder: dict, weight_dtype: str) -> io.NodeOutput:
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

        unet_path = resolve_selected_path("diffusion_models", unet_folder, "unet_folder", "unet_name")
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
