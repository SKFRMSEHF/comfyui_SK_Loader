// SK Loader tree selector enhancer
import { app } from "../../scripts/app.js";

function getTree(widget) {
  return (widget?.extra && widget.extra.sk_tree) || widget?._sk_tree || null;
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
  return `${folder.replace(/^\\/+|\\/+$|^\\/+/, "")}/${file}`;
}

function attachTreeHandler(node) {
  if (!node.widgets?.length) return;
  for (const widget of node.widgets) {
    if (widget.type !== "combo") continue;
    const tree = getTree(widget);
    if (!tree) continue;

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
        return true; // swallow
      }
      if (prevMouseDown) return prevMouseDown.call(widget, e, pos, graphcanvas);
      return false;
    };
  }
}

app.registerExtension({
  name: "sk_loader.tree_selector",
  async nodeCreated(node) {
    attachTreeHandler(node);
  },
});
