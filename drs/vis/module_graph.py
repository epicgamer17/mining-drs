import os
import urllib.request
import urllib.error

def _node_id(path):
    return "root" if not path else path.replace(".", "_")

def _node_label(path, mod):
    return path.split(".")[-1] if path else type(mod).__name__

def _generate_mermaid(model, show_vars=False) -> str:
    lines = []
    lines.append("flowchart TD")

    module_index = {}
    for path, mod in model.named_modules():
        module_index[id(mod)] = (path, mod)

    tree = {}
    for path, mod in model.named_modules():
        mod_id = id(mod)
        if path:
            parts = path.rsplit(".", 1)
            parent_path = parts[0] if len(parts) > 1 else ""
            parent_id = id(dict(model.named_modules()).get(parent_path))
            tree.setdefault(parent_id, []).append(mod_id)
        else:
            tree.setdefault(0, []).append(mod_id)

    rendered_mods = set()

    def _render_children(parent_mod_id, depth=0):
        prefix = "    " * depth
        for child_id in tree.get(parent_mod_id, []):
            if child_id in rendered_mods:
                continue
            rendered_mods.add(child_id)
            path, mod = module_index[child_id]
            nid = _node_id(path)
            label = _node_label(path, mod)
            
            grandchild_ids = tree.get(child_id, [])

            if grandchild_ids:
                lines.append(f'{prefix}subgraph {nid}["<b>{_node_label(path, mod)}</b>"]')
                
                if show_vars and mod is not None:
                    var_names = list(mod._variables.keys())[:5]
                    if var_names:
                        var_label = f"<b>{_node_label(path, mod)}</b> vars"
                        var_label += "<br><i>" + "</i><br><i>".join(var_names[:5]) + "</i>"
                        if len(mod._variables) > 5:
                            var_label += f"<br><i>+{len(mod._variables) - 5} more</i>"
                        lines.append(f'{prefix}    {nid}_vars[/"{var_label}"\\]')
                        lines.append(f'{prefix}    style {nid}_vars fill:transparent,stroke-dasharray: 5 5')
                        
                _render_children(child_id, depth + 1)
                lines.append(f"{prefix}end")
            else:
                label = f"<b>{_node_label(path, mod)}</b>"
                if show_vars and mod is not None:
                    var_names = list(mod._variables.keys())[:5]
                    if var_names:
                        label += "<br><i>" + "</i><br><i>".join(var_names[:5]) + "</i>"
                        if len(mod._variables) > 5:
                            label += f"<br><i>+{len(mod._variables) - 5} more</i>"
                lines.append(f'{prefix}{nid}(["{label}"])')

    _render_children(0)

    seen_edges = set()
    for rpath, rmod in model.named_modules():
        for src_mod, dep_var in rmod._dependencies:
            src_info = module_index.get(id(src_mod))
            if src_info is None:
                continue
            spath, _ = src_info
            rdr_path = rpath if rpath else ""
            if not spath or not rdr_path:
                continue
            src_id = _node_id(spath)
            rdr_id = _node_id(rdr_path)
            if src_id == rdr_id or src_id == "root" or rdr_id == "root":
                continue
            edge_key = (src_id, rdr_id, dep_var.name)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                lines.append(f"    {src_id} -->|{dep_var.name}| {rdr_id}")

    for rpath, rmod in model.named_modules():
        for src_mod in rmod._flow_dependencies:
            src_info = module_index.get(id(src_mod))
            if src_info is None:
                continue
            spath, _ = src_info
            rdr_path = rpath if rpath else ""
            if not spath or not rdr_path:
                continue
            src_id = _node_id(spath)
            rdr_id = _node_id(rdr_path)
            if src_id == rdr_id or src_id == "root" or rdr_id == "root":
                continue
            edge_key = (src_id, rdr_id, "flow")
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                lines.append(f"    {src_id} ==>|flow| {rdr_id}")

    for rpath, rmod in model.named_modules():
        for src_mod in getattr(rmod, "_data_dependencies", []):
            src_info = module_index.get(id(src_mod))
            if src_info is None:
                continue
            spath, _ = src_info
            rdr_path = rpath if rpath else ""
            if not spath or not rdr_path:
                continue
            src_id = _node_id(spath)
            rdr_id = _node_id(rdr_path)
            if src_id == rdr_id or src_id == "root" or rdr_id == "root":
                continue
            edge_key = (src_id, rdr_id, "data")
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                lines.append(f"    {src_id} -.->|data| {rdr_id}")

    return "\n".join(lines)


def render_kroki_png(mermaid_code: str, output_path: str):
    url = "https://kroki.io/mermaid/png"
    data = mermaid_code.encode("utf-8")
    headers = {
        "Content-Type": "text/plain",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
        print(f"Saved mermaid graph png via Kroki to {output_path}")
    except Exception as e:
        print(f"Failed to render mermaid graph via Kroki: {e}")


def save_module_graph_report(model, path_prefix="module_graph", show_vars=True):
    png_path = f"{path_prefix}.png"
    md_path = f"{path_prefix}.md"

    # Generate mermaid code
    mermaid_code = _generate_mermaid(model, show_vars=show_vars)

    # Render PNG using Kroki
    render_kroki_png(mermaid_code, png_path)

    dep_lines = []
    flow_lines = []
    for rpath, rmod in model.named_modules():
        for src_mod, dep_var in rmod._dependencies:
            for spath, smod in model.named_modules():
                if smod is src_mod:
                    src_name = spath if spath else type(smod).__name__
                    rdr_name = rpath if rpath else type(rmod).__name__
                    if src_name != rdr_name:
                        dep_lines.append(
                            f"  - `{src_name}` → `{rdr_name}` reads `{dep_var.name}`"
                        )
                    break
        for src_mod in rmod._flow_dependencies:
            for spath, smod in model.named_modules():
                if smod is src_mod:
                    src_name = spath if spath else type(smod).__name__
                    rdr_name = rpath if rpath else type(rmod).__name__
                    if src_name != rdr_name:
                        flow_lines.append(f"  - `{src_name}` → `{rdr_name}` flow")
                    break

        for src_mod in getattr(rmod, "_data_dependencies", []):
            for spath, smod in model.named_modules():
                if smod is src_mod:
                    src_name = spath if spath else type(smod).__name__
                    rdr_name = rpath if rpath else type(rmod).__name__
                    if src_name != rdr_name:
                        flow_lines.append(f"  - `{src_name}` -.-> `{rdr_name}` data lookup")
                    break

    module_list = []
    for path, mod in model.named_modules():
        short = path.split(".")[-1] if path else type(mod).__name__
        v = list(mod._variables.keys())[:5]
        v_str = ", ".join(v) if v else "—"
        if len(mod._variables) > 5:
            v_str += f", +{len(mod._variables) - 5} more"
        module_list.append(
            f"| `{short}` | `{path if path else '(root)'}` | `{v_str}` |"
        )

    deps_section = ""
    if dep_lines:
        deps_section = f"""## Data Dependencies (persistent variable reads)

The following read-dependencies were recorded during the simulation. An arrow `A → B` means module B reads a variable owned by module A.

{chr(10).join(dep_lines)}

"""
    if flow_lines:
        deps_section += f"""## Data Flow (transient)

The following transient flow-edges were recorded during the simulation. An arrow `A → B` means module A returned a `drs.Flow` value that was passed as input to module B.

{chr(10).join(flow_lines)}

"""

    md_content = f"""# DRS Module Graph — {type(model).__name__}

> Generated automatically by `drs.vis.module_graph.save_module_graph_report`

## Module Hierarchy

| Name | Path | Variables |
|------|------|-----------|
{chr(10).join(module_list)}

## Flowchart

```mermaid
{mermaid_code}
```

{deps_section}## Visual Graph

![Module Graph]({os.path.basename(png_path)})
"""

    with open(md_path, "w",encoding="utf-8") as f:
        f.write(md_content)

    print(f"Saved module graph report to {md_path}")
    return md_path
