// SK Loader tree selector enhancer (ES module loaded by ComfyUI)
// Path mirrors common custom-node scripts usage.
import { app } from "../../scripts/app.js";

const TREE_SENTINEL = "[[SK_TREE::";

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
  return null;
}

function makeMenuItems(tree, onSelect) {
  return tree
    .filter((n) => n && typeof n === "object")
    .map((n) => {
      const hasChildren = Array.isArray(n.children) && n.children.length > 0;
      if (hasChildren) {
        return {
          content: n.label ?? "folder",
          submenu: {
            options: makeMenuItems(n.children, onSelect),
          },
        };
      }
      if (!n.value) {
        return null;
      }
      return {
        content: n.label ?? n.value?.file ?? "file",
        callback: () => onSelect(n.value),
      };
    })
    .filter(Boolean);
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
      const menuItems = makeMenuItems(tree, (val) => {
        const newVal = valueToPath(val);
        if (!newVal) return;
        widget.value = newVal;
        if (typeof widget.callback === "function") {
          widget.callback(newVal, this, pos, graphcanvas);
        }
        node.setDirtyCanvas(true, true);
      });
      if (menuItems && menuItems.length) {
        new LiteGraph.ContextMenu(menuItems, {
          event: e,
          className: "dark",
          title: widget.name || "Select",
        });
        return true; // swallow default combo
      }
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
