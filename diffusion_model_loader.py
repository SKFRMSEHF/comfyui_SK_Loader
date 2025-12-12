from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, _io

import os

import folder_paths


def list_dirs(folder_type: str) -> list[str]:
    dirs = set([""])
    for base in folder_paths.get_folder_paths(folder_type):
        for root, subdirs, _ in os.walk(base):
            for d in subdirs:
                rel = os.path.relpath(os.path.join(root, d), base)
                rel = rel.replace("\\", "/")
                dirs.add(rel)
    return sorted(dirs)


def list_files(folder_type: str, rel_dir: str) -> list[str]:
    files: list[str] = []
    allowed_ext = {".safetensors", ".ckpt", ".bin", ".pt", ".pth"}
    for base in folder_paths.get_folder_paths(folder_type):
        dir_path = os.path.join(base, rel_dir) if rel_dir else base
        if not os.path.isdir(dir_path):
            continue
        for root, _, fnames in os.walk(dir_path):
            for f in fnames:
                name_low = f.lower()
                ext = os.path.splitext(name_low)[1]
                if ext not in allowed_ext:
                    continue
                rel = os.path.relpath(os.path.join(root, f), dir_path)
                rel = rel.replace("\\", "/")
                files.append(rel if rel_dir == "" else os.path.join(rel_dir, rel).replace("\\", "/"))
    return sorted(set(files))


def build_folder_file_input(folder_id: str, file_id: str, folder_type: str, tooltip: str | None = None) -> _io.DynamicCombo.Input:
    options: list[_io.DynamicCombo.Option] = []
    for rel_dir in list_dirs(folder_type):
        file_options = list_files(folder_type, rel_dir)
        if not file_options:
            continue
        inputs = [io.Combo.Input(file_id, options=file_options, tooltip=tooltip)]
        options.append(_io.DynamicCombo.Option(rel_dir if rel_dir else "root", inputs))
    if not options:
        options.append(_io.DynamicCombo.Option("No files found", [io.Combo.Input(file_id, options=["<none>"], tooltip="No files found")]))
    return _io.DynamicCombo.Input(folder_id, options=options, tooltip="Select folder")


def resolve_selected_path(folder_type: str, selection: dict, folder_id: str, file_id: str) -> str:
    if not isinstance(selection, dict):
        raise ValueError("Invalid selection")
    rel_dir = selection.get(folder_id, "root")
    if rel_dir == "root":
        rel_dir = ""
    file_rel = selection.get(file_id)
    if not file_rel or file_rel == "<none>":
        raise FileNotFoundError("No file selected")
    rel_path = file_rel.replace("\\", "/")
    for base in folder_paths.get_folder_paths(folder_type):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve path for selection: {rel_path}")


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
