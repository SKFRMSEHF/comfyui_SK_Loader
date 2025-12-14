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
                build_file_input("ckpt", "checkpoints"),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
            ],
        )

    @classmethod
    def execute(cls, config_name: str, ckpt: dict | str) -> io.NodeOutput:
        import comfy.sd

        config_path = folder_paths.get_full_path("configs", config_name)
        ckpt_path = resolve_selected_path("checkpoints", ckpt, "ckpt", "ckpt")
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
                build_file_input("ckpt", "checkpoints", tooltip="The checkpoint (model) to load."),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
            ],
            description="Loads a diffusion model checkpoint, diffusion models are used to denoise latents.",
        )

    @classmethod
    def execute(cls, ckpt: dict | str) -> io.NodeOutput:
        import comfy.sd

        ckpt_path = resolve_selected_path("checkpoints", ckpt, "ckpt", "ckpt")
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
                build_file_input("ckpt", "checkpoints"),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
                io.ClipVision.Output(),
            ],
        )

    @classmethod
    def execute(cls, ckpt: dict | str) -> io.NodeOutput:
        import comfy.sd

        ckpt_path = resolve_selected_path("checkpoints", ckpt, "ckpt", "ckpt")
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
