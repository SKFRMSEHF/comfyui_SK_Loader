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


def _sanitize(rel_dir: str) -> str:
    return (rel_dir.replace("\\", "/") if rel_dir else "root").replace("/", "__")


def build_folder_file_input(folder_id: str, file_id: str, folder_type: str, tooltip: str | None = None) -> _io.DynamicCombo.Input:
    options: list[_io.DynamicCombo.Option] = []
    for rel_dir in list_dirs(folder_type):
        file_options = list_files(folder_type, rel_dir)
        if not file_options:
            continue
        child_id = f"{file_id}__{_sanitize(rel_dir)}"
        inputs = [io.Combo.Input(child_id, options=file_options, tooltip=tooltip)]
        options.append(_io.DynamicCombo.Option(rel_dir if rel_dir else "root", inputs))
    if not options:
        child_id = f"{file_id}__none"
        options.append(_io.DynamicCombo.Option("No files found", [io.Combo.Input(child_id, options=["<none>"], tooltip="No files found")]))
    return _io.DynamicCombo.Input(folder_id, options=options, tooltip="Select folder")


def resolve_selected_path(folder_type: str, selection: dict, folder_id: str, file_id: str) -> str:
    if not isinstance(selection, dict):
        raise ValueError("Invalid selection")
    rel_dir = selection.get(folder_id, "root")
    if rel_dir == "root":
        rel_dir = ""
    child_id = f"{file_id}__{_sanitize(rel_dir)}" if rel_dir != "" else f"{file_id}__root"
    file_rel = selection.get(child_id)
    if not file_rel or file_rel == "<none>":
        raise FileNotFoundError("No file selected")
    rel_path = file_rel.replace("\\", "/")
    for base in folder_paths.get_folder_paths(folder_type):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve path for selection: {rel_path}")


class LoraLoader(io.ComfyNode):
    CATEGORY = "SK Loader"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_LoraLoader",
            display_name="[SK] LoRA",
            category="SK Loader",
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model the LoRA will be applied to."),
                io.Clip.Input("clip", tooltip="The CLIP model the LoRA will be applied to."),
                build_folder_file_input("lora_folder", "lora_name", "loras", tooltip="The name of the LoRA."),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="How strongly to modify the diffusion model. This value can be negative.",
                ),
                io.Float.Input(
                    "strength_clip",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="How strongly to modify the CLIP model. This value can be negative.",
                ),
            ],
            outputs=[
                io.Model.Output(tooltip="The modified diffusion model."),
                io.Clip.Output(tooltip="The modified CLIP model."),
            ],
            description="LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together.",
        )

    @classmethod
    def _apply_lora(cls, model, clip, selection: dict, strength_model: float, strength_clip: float):
        import comfy.sd
        import comfy.utils

        if strength_model == 0 and strength_clip == 0:
            return model, clip

        lora_path = resolve_selected_path("loras", selection, "lora_folder", "lora_name")
        loaded = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return comfy.sd.load_lora_for_models(model, clip, loaded, strength_model, strength_clip)

    @classmethod
    def execute(cls, model, clip, lora_folder: dict, strength_model: float, strength_clip: float) -> io.NodeOutput:
        model_lora, clip_lora = cls._apply_lora(model, clip, lora_folder, strength_model, strength_clip)
        return io.NodeOutput(model_lora, clip_lora)


class LoraLoaderModelOnly(LoraLoader):
    CATEGORY = "SK Loader"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_LoraLoaderModelOnly",
            display_name="[SK] LoRA (Model Only)",
            category="SK Loader",
            inputs=[
                io.Model.Input("model"),
                build_folder_file_input("lora_folder", "lora_name", "loras"),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, lora_folder: dict, strength_model: float) -> io.NodeOutput:
        model_lora, _ = cls._apply_lora(model, None, lora_folder, strength_model, 0)
        return io.NodeOutput(model_lora)


class LoraExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoraLoader,
            LoraLoaderModelOnly,
        ]


async def comfy_entrypoint() -> LoraExtension:
    return LoraExtension()
