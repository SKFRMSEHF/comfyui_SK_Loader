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


class CheckpointLoader(io.ComfyNode):
    CATEGORY = "SK Loader/Advanced"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_CheckpointLoader",
            display_name="[SK] Checkpoint (Config, Deprecated)",
            category="SK Loader/Advanced",
            inputs=[
                io.Combo.Input("config_name", options=folder_paths.get_filename_list("configs")),
                build_folder_file_input("ckpt_folder", "ckpt_name", "checkpoints"),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
            ],
        )

    @classmethod
    def execute(cls, config_name: str, ckpt_folder: dict) -> io.NodeOutput:
        import comfy.sd

        config_path = folder_paths.get_full_path("configs", config_name)
        ckpt_path = resolve_selected_path("checkpoints", ckpt_folder, "ckpt_folder", "ckpt_name")
        loaded = comfy.sd.load_checkpoint(
            config_path,
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(*loaded)


class CheckpointLoaderSimple(io.ComfyNode):
    CATEGORY = "SK Loader"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_CheckpointLoaderSimple",
            display_name="[SK] Checkpoint",
            category="SK Loader",
            inputs=[
                build_folder_file_input("ckpt_folder", "ckpt_name", "checkpoints", tooltip="The checkpoint (model) to load."),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
            ],
            description="Loads a diffusion model checkpoint, diffusion models are used to denoise latents.",
        )

    @classmethod
    def execute(cls, ckpt_folder: dict) -> io.NodeOutput:
        import comfy.sd

        ckpt_path = resolve_selected_path("checkpoints", ckpt_folder, "ckpt_folder", "ckpt_name")
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(*out[:3])


class unCLIPCheckpointLoader(io.ComfyNode):
    CATEGORY = "SK Loader"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_unCLIPCheckpointLoader",
            display_name="[SK] unCLIP Checkpoint",
            category="SK Loader",
            inputs=[
                build_folder_file_input("ckpt_folder", "ckpt_name", "checkpoints"),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
                io.ClipVision.Output(),
            ],
        )

    @classmethod
    def execute(cls, ckpt_folder: dict) -> io.NodeOutput:
        import comfy.sd

        ckpt_path = resolve_selected_path("checkpoints", ckpt_folder, "ckpt_folder", "ckpt_name")
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            output_clipvision=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(*out)


class LoaderExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CheckpointLoader,
            CheckpointLoaderSimple,
            unCLIPCheckpointLoader,
        ]


async def comfy_entrypoint() -> LoaderExtension:
    return LoaderExtension()
