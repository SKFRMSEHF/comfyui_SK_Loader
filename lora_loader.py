from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, _io

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


def build_lora_slot_inputs(idx: int) -> list:
    """Return inputs for a single LoRA slot (enable + select + strengths)."""
    prefix = f"lora_{idx}"
    label = f"LoRA #{idx}"
    return [
        io.Bool.Input(f"{prefix}_enabled", default=False, tooltip=f"Enable {label}"),
        build_folder_file_input(f"{prefix}_folder", f"{prefix}_name", "loras", tooltip=f"Select {label}"),
        io.Float.Input(
            f"{prefix}_strength_model",
            default=1.0,
            min=-100.0,
            max=100.0,
            step=0.01,
            tooltip=f"{label} model strength",
        ),
        io.Float.Input(
            f"{prefix}_strength_clip",
            default=1.0,
            min=-100.0,
            max=100.0,
            step=0.01,
            tooltip=f"{label} CLIP strength",
        ),
    ]


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


class PowerLoraLoader(io.ComfyNode):
    CATEGORY = "SK Loader"
    NUM_SLOTS = 5

    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs: list = [
            io.Model.Input("model", tooltip="Diffusion model to apply multiple LoRAs onto."),
            io.Clip.Input("clip", tooltip="CLIP model to apply multiple LoRAs onto."),
        ]
        for idx in range(1, cls.NUM_SLOTS + 1):
            inputs.extend(build_lora_slot_inputs(idx))
        return io.Schema(
            node_id="SK_PowerLoraLoader",
            display_name="[SK] Power LoRA Loader",
            category="SK Loader",
            inputs=inputs,
            outputs=[
                io.Model.Output(tooltip="The modified diffusion model."),
                io.Clip.Output(tooltip="The modified CLIP model."),
            ],
            description="Apply multiple LoRAs in order, each with enable toggles and strengths.",
        )

    @classmethod
    def execute(cls, model, clip, **kwargs) -> io.NodeOutput:
        import comfy.sd
        import comfy.utils

        model_out, clip_out = model, clip

        for idx in range(1, cls.NUM_SLOTS + 1):
            enabled = kwargs.get(f"lora_{idx}_enabled", False)
            if not enabled:
                continue

            selection = kwargs.get(f"lora_{idx}_folder")
            if not isinstance(selection, dict):
                continue

            strength_model = float(kwargs.get(f"lora_{idx}_strength_model", 1.0))
            strength_clip = float(kwargs.get(f"lora_{idx}_strength_clip", 1.0))
            if strength_model == 0 and strength_clip == 0:
                continue

            try:
                lora_path = resolve_selected_path("loras", selection, f"lora_{idx}_folder", f"lora_{idx}_name")
            except FileNotFoundError:
                continue

            loaded = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_out, clip_out = comfy.sd.load_lora_for_models(
                model_out,
                clip_out,
                loaded,
                strength_model,
                strength_clip if clip_out is not None else 0,
            )

        return io.NodeOutput(model_out, clip_out)


class LoraExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoraLoader,
            LoraLoaderModelOnly,
            PowerLoraLoader,
        ]


async def comfy_entrypoint() -> LoraExtension:
    return LoraExtension()
