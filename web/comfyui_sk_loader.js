// SK Loader tree selector enhancer (ES module loaded by ComfyUI)
// Path mirrors common custom-node scripts usage.
import { app } from "../../scripts/app.js";

const TREE_SENTINEL = "[[SK_TREE::";
const STYLE_ID = "sk-loader-tree-style";

let activeTreeMenu = null;

function deriveTreeFromOptions(widget) {
  const raw =
    widget?.options?.values ??
    widget?.options_values ??
    widget?.options ??
    widget?.combo_items ??
    widget?.items;
  const values = Array.isArray(raw) ? raw : typeof raw === "object" && raw ? Object.values(raw) : null;
  if (!values) return null;

  const root = [];

  const ensureFolder = (children, label) => {
    let existing = children.find((c) => c.label === label && Array.isArray(c.children));
    if (!existing) {
      existing = { label, value: null, children: [] };
      children.push(existing);
    }
    return existing.children;
  };

  for (const val of values) {
    if (typeof val !== "string") continue;
    const normalized = val.replace(/\\/g, "/");
    const parts = normalized.split("/");
    if (parts.length === 1) {
      root.push({ label: normalized, value: normalized, children: [] });
      continue;
    }
    let children = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isLeaf = i === parts.length - 1;
      if (isLeaf) {
        children.push({ label: part || normalized, value: normalized, children: [] });
      } else {
        children = ensureFolder(children, part || "root");
      }
    }
  }
  return root.length ? root : null;
}

function getTree(widget) {
  const metaTree = (widget && widget.extra && widget.extra.sk_tree) || widget?.metadata?.sk_tree || widget?._sk_tree;
  if (metaTree) return metaTree;
  const tip = widget?.tooltip;
  if (typeof tip === "string") {
    const idx = tip.indexOf(TREE_SENTINEL);
    if (idx >= 0) {
      const json = tip.substring(idx + TREE_SENTINEL.length).split("]]")[0];
      try {
        return JSON.parse(json);
      } catch (e) {
        console.warn("SK Loader: failed to parse tree JSON from tooltip", e);
      }
    }
  }
  return deriveTreeFromOptions(widget);
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .sk-tree-menu{position:fixed;z-index:10000;background:#111;border:1px solid #363636;border-radius:8px;box-shadow:0 10px 28px rgba(0,0,0,0.45);color:#f0f0f0;max-height:520px;min-width:260px;max-width:520px;overflow:auto;font-size:12px;font-family:var(--comfy-font,Inter,system-ui,sans-serif);padding:4px 0;}
    .sk-tree-list{min-width:260px;}
    .sk-tree-folder>summary{list-style:none;cursor:pointer;padding:6px 8px 6px 10px;margin:0;user-select:none;color:#f0f0f0;}
    .sk-tree-folder summary::-webkit-details-marker{display:none;}
    .sk-tree-folder summary::before{content:">";display:inline-block;width:12px;color:#8a8a8a;transition:transform 0.12s ease;}
    .sk-tree-folder[open] summary::before{transform:rotate(90deg);}
    .sk-tree-folder>summary:hover{background:#1d1d1d;}
    .sk-tree-leaf{cursor:pointer;padding:6px 8px 6px 26px;margin:0;user-select:none;white-space:nowrap;color:#f0f0f0;}
    .sk-tree-leaf:hover{background:#1d1d1d;}
    .sk-tree-selected{background:#244466;color:#fff;}
  `;
  document.head.appendChild(style);
}

function valueToPath(val) {
  if (typeof val === "string") return val;
  if (!val || typeof val !== "object") return null;
  const file = val.file || val.path;
  if (!file) return null;
  const folder = val.folder;
  if (!folder || folder === "root") return file;
  const cleanFolder = folder.replace(/^[/\\]+|[/\\]+$/g, "");
  return `${cleanFolder}/${file}`;
}

function normalizePath(path) {
  return (path || "").toString().replace(/\\/g, "/");
}

function labelForNode(n) {
  if (!n || typeof n !== "object") return "item";
  return n.label ?? n.value?.file ?? "item";
}

function buildTreeNode(node, depth, onSelect, currentValue) {
  if (!node || typeof node !== "object") return null;
  const hasChildren = Array.isArray(node.children) && node.children.length > 0;
  const label = labelForNode(node);

  if (hasChildren) {
    const details = document.createElement("details");
    details.className = "sk-tree-folder";
    details.open = depth < 1;

    const summary = document.createElement("summary");
    summary.textContent = label;
    summary.style.paddingLeft = `${10 + depth * 14}px`;
    details.appendChild(summary);

    for (const child of node.children) {
      const childEl = buildTreeNode(child, depth + 1, onSelect, currentValue);
      if (childEl) details.appendChild(childEl);
    }
    if (details.childElementCount <= 1) return null; // only summary, no usable children
    return details;
  }

  const path = valueToPath(node.value);
  if (!path) return null;
  const normalized = normalizePath(path);
  const leaf = document.createElement("div");
  leaf.className = "sk-tree-leaf";
  leaf.textContent = label;
  leaf.style.paddingLeft = `${26 + depth * 14}px`;
  if (normalized === currentValue) {
    leaf.classList.add("sk-tree-selected");
  }
  leaf.addEventListener("pointerdown", (ev) => {
    ev.stopPropagation();
    ev.preventDefault();
    onSelect(node.value);
  });
  return leaf;
}

function buildTreeList(tree, onSelect, currentValue) {
  const list = document.createElement("div");
  list.className = "sk-tree-list";
  for (const node of tree) {
    const el = buildTreeNode(node, 0, onSelect, currentValue);
    if (el) list.appendChild(el);
  }
  return list.childElementCount > 0 ? list : null;
}

function closeActiveMenu() {
  if (!activeTreeMenu) return;
  const { menu, dismiss, onKey } = activeTreeMenu;
  document.removeEventListener("pointerdown", dismiss, true);
  document.removeEventListener("keydown", onKey);
  menu.remove();
  activeTreeMenu = null;
}

function openTreeMenu(widget, tree, event, node, graphcanvas) {
  if (!Array.isArray(tree) || !tree.length) return false;
  ensureStyles();
  if (typeof LiteGraph !== "undefined" && LiteGraph.closeAllContextMenus) {
    LiteGraph.closeAllContextMenus();
  }
  closeActiveMenu();

  const currentValue = normalizePath(valueToPath(widget.value) || widget.value || "");
  const menu = document.createElement("div");
  menu.className = "sk-tree-menu";

  const list = buildTreeList(
    tree,
    (val) => {
      const newVal = valueToPath(val);
      if (!newVal) return;
      widget.value = newVal;
      if (typeof widget.callback === "function") {
        widget.callback(newVal, widget, event, graphcanvas);
      }
      node.setDirtyCanvas(true, true);
      closeActiveMenu();
    },
    currentValue
  );

  if (!list) {
    return false;
  }

  menu.appendChild(list);
  document.body.appendChild(menu);

  let left = event?.clientX ?? 0;
  let top = event?.clientY ?? 0;
  const rect = menu.getBoundingClientRect();
  if (left + rect.width > window.innerWidth) {
    left = Math.max(8, window.innerWidth - rect.width - 12);
  }
  if (top + rect.height > window.innerHeight) {
    top = Math.max(8, window.innerHeight - rect.height - 12);
  }
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;

  const dismiss = (ev) => {
    if (!menu.contains(ev.target)) {
      closeActiveMenu();
    }
  };
  const onKey = (ev) => {
    if (ev.key === "Escape") {
      closeActiveMenu();
    }
  };

  document.addEventListener("pointerdown", dismiss, true);
  document.addEventListener("keydown", onKey);
  menu.addEventListener("pointerdown", (ev) => ev.stopPropagation());
  activeTreeMenu = { menu, dismiss, onKey };

  return true;
}

function attachTreeHandler(node) {
  if (!node.widgets || !node.widgets.length) return;
  for (const widget of node.widgets) {
    if (!widget || widget.type !== "combo") continue;
    if (widget._sk_tree_bound) continue;
    const tree = getTree(widget);
    if (!tree) continue;
    widget._sk_tree_bound = true;

    const prevMouseDown = widget.onMouseDown;
    widget.onMouseDown = function (e, pos, graphcanvas) {
      const handled = openTreeMenu(widget, tree, e, node, graphcanvas);
      if (handled) return true;
      if (prevMouseDown) return prevMouseDown.call(widget, e, pos, graphcanvas);
      return false;
    };
  }
}

function attachExistingNodes() {
  const nodes = app?.graph?._nodes;
  if (!nodes) return;
  for (const n of nodes) {
    attachTreeHandler(n);
  }
}

app.registerExtension({
  name: "sk_loader.tree_selector",
  setup() {
    // Attach once after initial graph load.
    setTimeout(attachExistingNodes, 50);
    setTimeout(attachExistingNodes, 250);
  },
  nodeCreated(node) {
    attachTreeHandler(node);
  },
});
