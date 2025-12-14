import json
import os
from typing import Any, Iterable

import folder_paths


# File extensions we consider as model artifacts for the loaders.
ALLOWED_EXT = {".safetensors", ".ckpt", ".bin", ".pt", ".pth"}

# Sentinel marker to tuck tree JSON into tooltips as a fallback transport.
TREE_SENTINEL = "[[SK_TREE::"


def list_dirs(folder_type: str) -> list[str]:
    """Return all subdirectories (relative) under the registered folder type."""
    dirs = set([""])
    for base in folder_paths.get_folder_paths(folder_type):
        for root, subdirs, _ in os.walk(base):
            for d in subdirs:
                rel = os.path.relpath(os.path.join(root, d), base)
                rel = rel.replace("\\", "/")
                dirs.add(rel)
    return sorted(dirs)


def list_files(folder_type: str, rel_dir: str) -> list[str]:
    """Return files under a relative directory for the given folder type."""
    files: list[str] = []
    for base in folder_paths.get_folder_paths(folder_type):
        dir_path = os.path.join(base, rel_dir) if rel_dir else base
        if not os.path.isdir(dir_path):
            continue
        for root, _, fnames in os.walk(dir_path):
            for f in fnames:
                name_low = f.lower()
                ext = os.path.splitext(name_low)[1]
                if ext not in ALLOWED_EXT:
                    continue
                rel = os.path.relpath(os.path.join(root, f), dir_path)
                rel = rel.replace("\\", "/")
                files.append(rel if rel_dir == "" else os.path.join(rel_dir, rel).replace("\\", "/"))
    return sorted(set(files))


def sanitize_rel_dir(rel_dir: str) -> str:
    """Match the sanitize logic used by the original DynamicCombo inputs."""
    return (rel_dir.replace("\\", "/") if rel_dir else "root").replace("/", "__")


def _ensure_branch(children: list[dict[str, Any]], parts: Iterable[str]) -> list[dict[str, Any]]:
    """Ensure nested folder nodes exist for the given path parts and return the leaf children list."""
    current_children = children
    for part in parts:
        if part == "":
            continue
        existing = next((c for c in current_children if c.get("label") == part and c.get("children") is not None), None)
        if existing is None:
            existing = {"label": part, "value": None, "children": []}
            current_children.append(existing)
        current_children = existing["children"]
    return current_children


def build_tree(folder_type: str, file_id: str, extra_roots: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Build a folder/file tree for the given Comfy folder type."""
    tree: list[dict[str, Any]] = []

    for base in folder_paths.get_folder_paths(folder_type):
        base_label = os.path.basename(base) or folder_type
        base_node: dict[str, Any] = {"label": base_label, "value": None, "children": []}

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
                folder_val = rel_root if rel_root else "root"
                child_id = f"{file_id}__{sanitize_rel_dir(rel_root)}"
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

    if extra_roots:
        tree.extend(extra_roots)
    return tree


def resolve_selected_path(folder_type: str, selection: dict, folder_id: str, file_id: str) -> str:
    """Resolve a selection map (combo or tree UI) to an absolute path under the given folder type."""
    if not isinstance(selection, dict):
        raise ValueError("Invalid selection")

    def _valid(val: Any) -> bool:
        return isinstance(val, str) and val not in ("", "<none>")

    folder_sel = selection.get(folder_id)
    rel_dir = "" if folder_sel in (None, "root", "") else str(folder_sel)
    file_rel: str | None = selection.get(file_id) if folder_sel is None else None

    if folder_sel is not None:
        child_id = f"{file_id}__{sanitize_rel_dir(rel_dir)}"
        file_rel = selection.get(child_id) or selection.get(file_id)

    if not _valid(file_rel):
        for key, val in selection.items():
            if not key.startswith(f"{file_id}__"):
                continue
            if not _valid(val):
                continue
            file_rel = val
            if folder_sel is None:
                suffix = key[len(f"{file_id}__") :]
                if suffix not in ("root", ""):
                    rel_dir = suffix.replace("__", "/")
            break

    if not _valid(file_rel):
        raise FileNotFoundError("No file selected")

    rel_path = file_rel.replace("\\", "/")
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path

    for base in folder_paths.get_folder_paths(folder_type):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not resolve path for selection: {rel_path}")


def attach_tree_metadata(input_obj: Any, tree: list[dict[str, Any]], tooltip: str | None = None) -> Any:
    """Attach tree data to the input in several ways so the frontend can pick it up."""
    tooltip = tooltip or "Select folder"
    encoded = json.dumps(tree, separators=(",", ":"))
    sentinel = f"{TREE_SENTINEL}{encoded}]]"
    try:
        # Keep the human tooltip on the first line, stash tree data after sentinel.
        input_obj.tooltip = f"{tooltip}\n{sentinel}"
    except Exception:
        pass

    payload = {"sk_tree": tree}
    for attr in ("extra", "metadata", "ui"):
        try:
            setattr(input_obj, attr, payload)
            break
        except Exception:
            continue
    try:
        setattr(input_obj, "_sk_tree", tree)
    except Exception:
        pass
    return input_obj
