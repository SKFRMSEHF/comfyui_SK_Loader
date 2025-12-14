import os
from typing import Any

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

import folder_paths
from .tree_utils import ALLOWED_EXT, attach_tree_metadata, list_dirs, list_files, sanitize_rel_dir, _ensure_branch

BUILTIN_VAES = ["pixel_space", "taesd", "taesdxl", "taesd3", "taef1"]


def build_vae_tree(file_id: str) -> list[dict[str, Any]]:
    """Build a combined tree for VAE/built-in sources with stable child IDs."""
    tree: list[dict[str, Any]] = [
        {
            "label": "builtins",
            "value": None,
            "children": [
                {
                    "label": name,
                    "value": {"folder": "builtins", "file": name, "child_id": f"{file_id}__builtins"},
                    "children": [],
                }
                for name in BUILTIN_VAES
            ],
        }
    ]

    for folder_type in ("vae", "vae_approx"):
        for base in folder_paths.get_folder_paths(folder_type):
            base_label = os.path.basename(base) or folder_type
            base_node: dict[str, Any] = {"label": f"{folder_type}:{base_label}", "value": None, "children": []}

            for root, _, fnames in os.walk(base):
                rel_root = os.path.relpath(root, base)
                if rel_root in (".", ""):
                    rel_root = ""
                rel_root = rel_root.replace("\\", "/")

                target_children = _ensure_branch(base_node["children"], rel_root.split("/"))

                for fname in fnames:
                    ext = os.path.splitext(fname.lower())[1]
                    if ext not in ALLOWED_EXT:
                        continue
                    rel_path = fname if rel_root == "" else f"{rel_root}/{fname}"
                    folder_val = f"{folder_type}/{rel_root}" if rel_root else folder_type
                    child_id = f"{file_id}__{sanitize_rel_dir(folder_val)}"
                    target_children.append(
                        {
                            "label": fname,
                            "value": {
                                "folder": folder_val,
                                "file": rel_path.replace("\\", "/"),
                                "child_id": child_id,
                            },
                            "children": [],
                        }
                    )

            if base_node["children"]:
                tree.append(base_node)

    return tree


def build_vae_input(input_id: str) -> io.Combo.Input:
    """Single select with tree metadata; options = builtins + all vae/vae_approx files."""
    options: list[str] = []

    # built-in options
    options.extend(BUILTIN_VAES)

    for folder_type in ("vae", "vae_approx"):
        for rel_path in list_files(folder_type, ""):
            options.append(f"{folder_type}/{rel_path}")

    if not options:
        options = ["<none>"]

    combo = io.Combo.Input(input_id, options=sorted(set(options)), tooltip="Select VAE")
    tree = build_vae_tree(input_id)
    return attach_tree_metadata(combo, tree, tooltip="Select VAE")


def resolve_selected_path(selection: dict | str, folder_id: str = "vae_folder", file_id: str = "vae_name") -> str:
    """Resolve tree-aware single select or legacy two-combo selection to a usable path or builtin."""
    if isinstance(selection, str):
        rel = selection.strip()
        if rel in BUILTIN_VAES:
            return rel
        rel = rel.replace("\\", "/")
        if os.path.isabs(rel) and os.path.exists(rel):
            return rel
        for folder_type in ("vae", "vae_approx"):
            rel_no_prefix = rel[len(folder_type) + 1 :] if rel.startswith(f"{folder_type}/") else rel
            for candidate_rel in (rel_no_prefix, rel):
                for base in folder_paths.get_folder_paths(folder_type):
                    candidate = os.path.join(base, candidate_rel)
                    if os.path.exists(candidate):
                        return candidate
        return rel  # last resort

    if not isinstance(selection, dict):
        raise ValueError("Invalid selection")
    folder_sel = selection.get(folder_id, "builtins")
    file_rel = selection.get(file_id)

    def _valid(val: Any) -> bool:
        return isinstance(val, str) and val not in ("", "<none>")

    if folder_sel == "builtins":
        file_rel = selection.get(f"{file_id}__builtins") or file_rel
    elif folder_sel:
        child_id = f"{file_id}__{sanitize_rel_dir(folder_sel)}"
        file_rel = selection.get(child_id) or file_rel

    if not _valid(file_rel):
        for k, v in selection.items():
            if not k.startswith(f"{file_id}__"):
                continue
            if not _valid(v):
                continue
            suffix = k[len(f"{file_id}__") :]
            folder_sel = "builtins" if suffix == "builtins" else suffix.replace("__", "/")
            file_rel = v
            break

    if not _valid(file_rel):
        return "pixel_space"

    if folder_sel in ("builtins", None, ""):
        return file_rel  # handled separately

    parts = str(folder_sel).split("/", 1)
    folder_type = parts[0]
    rel_path = file_rel.replace("\\", "/")
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    for base in folder_paths.get_folder_paths(folder_type):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return file_rel  # last resort: return relative path


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
                build_vae_input("vae"),
            ],
            outputs=[
                io.Vae.Output(),
            ],
        )

    # TODO: scale factor?
    @classmethod
    def execute(cls, vae: dict | str) -> io.NodeOutput:
        import torch
        import comfy.sd
        import comfy.utils

        resolved = resolve_selected_path(vae, "vae_folder", "vae_name")

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
