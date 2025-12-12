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


def build_vae_input(folder_id: str, file_id: str) -> _io.DynamicCombo.Input:
    options: list[_io.DynamicCombo.Option] = []

    # built-in options
    builtins = ["pixel_space", "taesd", "taesdxl", "taesd3", "taef1"]
    options.append(_io.DynamicCombo.Option("builtins", [io.Combo.Input(f"{file_id}__builtins", options=builtins, tooltip="Built-in VAE/TAES")]))

    for folder_type in ("vae", "vae_approx"):
        for rel_dir in list_dirs(folder_type):
            file_options = list_files(folder_type, rel_dir)
            if not file_options:
                continue
            label = f"{folder_type}/{rel_dir}" if rel_dir else folder_type
            child_id = f"{file_id}__{_sanitize(label)}"
            inputs = [io.Combo.Input(child_id, options=file_options, tooltip="Select VAE file")]
            options.append(_io.DynamicCombo.Option(label, inputs))
    if not options:
        options.append(_io.DynamicCombo.Option("No files found", [io.Combo.Input(f"{file_id}__none", options=["<none>"], tooltip="No files found")]))
    return _io.DynamicCombo.Input(folder_id, options=options, tooltip="Select folder")


def resolve_selected_path(selection: dict, folder_id: str, file_id: str) -> str:
    if not isinstance(selection, dict):
        raise ValueError("Invalid selection")
    folder_sel = selection.get(folder_id, "builtins")
    file_rel = None

    # Primary lookups
    if folder_sel == "builtins":
        file_rel = selection.get(f"{file_id}__builtins") or selection.get(file_id)
    elif folder_sel:
        child_id = f"{file_id}__{_sanitize(folder_sel)}"
        file_rel = selection.get(child_id) or selection.get(file_id)

    # Fallback: pick first valid file entry
    if file_rel in (None, "<none>"):
        for k, v in selection.items():
            if k.startswith(f"{file_id}__") and isinstance(v, str) and v not in ("", "<none>"):
                file_rel = v
                if not folder_sel or folder_sel == "builtins":
                    suffix = k[len(f"{file_id}__"):]
                    folder_sel = "builtins" if suffix == "builtins" else suffix.replace("__", "/")
                break

    if not file_rel or file_rel == "<none>":
        file_rel = "pixel_space"
        folder_sel = "builtins"

    if folder_sel == "builtins":
        return file_rel  # handled separately

    parts = folder_sel.split("/", 1)
    folder_type = parts[0]
    rel_path = file_rel.replace("\\", "/")
    for base in folder_paths.get_folder_paths(folder_type):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return file_rel  # last resort: return relative path


def _sanitize(rel_dir: str) -> str:
    return (rel_dir.replace("\\", "/") if rel_dir else "root").replace("/", "__")


class VAELoader(io.ComfyNode):
    CATEGORY = "SK Loader"
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]

    @staticmethod
    def load_taesd(name):
        import torch
        import comfy.utils

        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SK_VAELoader",
            display_name="[SK] VAE",
            category="SK Loader",
            inputs=[
                build_vae_input("vae_folder", "vae_name"),
            ],
            outputs=[
                io.Vae.Output(),
            ],
        )

    # TODO: scale factor?
    @classmethod
    def execute(cls, vae_folder: dict) -> io.NodeOutput:
        import torch
        import comfy.sd
        import comfy.utils

        resolved = resolve_selected_path(vae_folder, "vae_folder", "vae_name")

        # Check if it's a builtin VAE
        if resolved in ("pixel_space", "taesd", "taesdxl", "taesd3", "taef1"):
            if resolved == "pixel_space":
                sd = {"pixel_space_vae": torch.tensor(1.0)}
            elif resolved in cls.image_taes:
                sd = cls.load_taesd(resolved)
            else:
                raise FileNotFoundError(f"Unknown builtin VAE: {resolved}")
        else:
            # Load VAE from file path
            sd = comfy.utils.load_torch_file(resolved)

        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return io.NodeOutput(vae)


class VAEExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VAELoader,
        ]


async def comfy_entrypoint() -> VAEExtension:
    return VAEExtension()
