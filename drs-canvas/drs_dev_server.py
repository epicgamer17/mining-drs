import http.server
import json
import os
import sys
import traceback

# Add the workspace root to sys.path so we can import 'drs'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(ROOT_DIR)
sys.path.append(WORKSPACE_ROOT)

# Default initial nodes and edges: Concentrator model
DEFAULT_NODES = [
  {
    "id": "mine",
    "type": "extractionNode",
    "position": { "x": 80, "y": 220 },
    "data": {
      "label": "Concentrator Mine Face",
      "class": "ConcentratorMineFace",
      "variables": {
        "active_parcel_initial_mass": { "class": "Variable", "value": 34975.28 },
        "active_parcel_ore_fraction": { "class": "Variable", "value": 0.7 },
        "cumulative_extracted_mass": { "class": "Level", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 0.0 },
        "parcel_extracted_mass": { "class": "Level", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 0.0 }
      }
    }
  },
  {
    "id": "fleet",
    "type": "factoryNode",
    "position": { "x": 340, "y": 200 },
    "data": {
      "label": "Continuous Fleet Logistics",
      "class": "ContinuousFleetLogistics",
      "variables": {
        "stockpile2_routing_fraction": { "class": "Variable", "value": 0.0 }
      }
    }
  },
  {
    "id": "ore1_stock",
    "type": "bufferNode",
    "position": { "x": 620, "y": 100 },
    "data": {
      "label": "Ore 1 Stockpile",
      "class": "Stockpile",
      "variables": {
        "current_mass": { "class": "Level", "value": 42000.0, "lower_threshold": 0.0, "upper_threshold": 60000.0, "rate": 0.0 },
        "contained_ore_fraction_mass": { "class": "Level", "value": 12600.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 0.0 },
        "actual_outflow_rate": { "class": "Variable", "value": 0.0 }
      }
    }
  },
  {
    "id": "ore2_stock",
    "type": "bufferNode",
    "position": { "x": 620, "y": 320 },
    "data": {
      "label": "Ore 2 Stockpile",
      "class": "Stockpile",
      "variables": {
        "current_mass": { "class": "Level", "value": 18000.0, "lower_threshold": 0.0, "upper_threshold": 60000.0, "rate": 0.0 },
        "contained_ore_fraction_mass": { "class": "Level", "value": 5400.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 0.0 },
        "actual_outflow_rate": { "class": "Variable", "value": 0.0 }
      }
    }
  },
  {
    "id": "plant",
    "type": "factoryNode",
    "position": { "x": 920, "y": 220 },
    "data": {
      "label": "Concentrator Plant",
      "class": "ConcentratorPlant",
      "variables": {
        "cumulative_milled_mass": { "class": "Level", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 0.0 }
      }
    }
  },
  {
    "id": "controller",
    "type": "factoryNode",
    "position": { "x": 340, "y": 440 },
    "data": {
      "label": "Concentrator Controller",
      "class": "ConcentratorController",
      "variables": {
        "active_operating_mode": { "class": "Variable", "value": "OperatingMode.MODE_A" },
        "total_system_ore_mass": { "class": "Level", "value": 60000.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 0.0 },
        "current_campaign_duration": { "class": "Timer", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 1.0 },
        "current_contingency_duration": { "class": "Timer", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 1.0 },
        "cumulative_time_mode_a": { "class": "Timer", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 1.0 },
        "cumulative_time_mode_b": { "class": "Timer", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 1.0 },
        "cumulative_time_shutdown": { "class": "Timer", "value": 0.0, "lower_threshold": "-Infinity", "upper_threshold": "Infinity", "rate": 1.0 },
        "target_mine_mass_rate": { "class": "Variable", "value": 0.0 },
        "target_stock1_outflow_rate": { "class": "Variable", "value": 0.0 },
        "target_stock2_outflow_rate": { "class": "Variable", "value": 0.0 }
      }
    }
  }
]

DEFAULT_EDGES = [
  {
    "id": "e-flow-mine-fleet",
    "source": "mine",
    "sourceHandle": "flow-out",
    "target": "fleet",
    "targetHandle": "flow-in",
    "style": { "stroke": "#3b82f6", "strokeWidth": 4 }
  },
  {
    "id": "e-flow-fleet-ore1",
    "source": "fleet",
    "sourceHandle": "flow-out",
    "target": "ore1_stock",
    "targetHandle": "flow-in",
    "style": { "stroke": "#3b82f6", "strokeWidth": 4 }
  },
  {
    "id": "e-flow-fleet-ore2",
    "source": "fleet",
    "sourceHandle": "flow-out",
    "target": "ore2_stock",
    "targetHandle": "flow-in",
    "style": { "stroke": "#3b82f6", "strokeWidth": 4 }
  },
  {
    "id": "e-flow-ore1-plant",
    "source": "ore1_stock",
    "sourceHandle": "flow-out",
    "target": "plant",
    "targetHandle": "flow-in",
    "style": { "stroke": "#3b82f6", "strokeWidth": 4 }
  },
  {
    "id": "e-flow-ore2-plant",
    "source": "ore2_stock",
    "sourceHandle": "flow-out",
    "target": "plant",
    "targetHandle": "flow-in",
    "style": { "stroke": "#3b82f6", "strokeWidth": 4 }
  },
  {
    "id": "e-read-ore1-controller",
    "source": "ore1_stock",
    "sourceHandle": "read-out",
    "target": "controller",
    "targetHandle": "read-in",
    "style": { "stroke": "#f59e0b", "strokeWidth": 1.5 }
  },
  {
    "id": "e-read-ore2-controller",
    "source": "ore2_stock",
    "sourceHandle": "read-out",
    "target": "controller",
    "targetHandle": "read-in",
    "style": { "stroke": "#f59e0b", "strokeWidth": 1.5 }
  }
]


def react_flow_to_drs_flat(nodes, edges):
    """Translate React Flow node/edge list format into flat DRS JSON schema."""
    drs_nodes = []
    
    # Identify the top-level pipeline module name. 
    # Usually, a node with path/id equal to "" or "root" serves as parent.
    parent_exists = any(n.get("id") == "" or n.get("id") == "root" for n in nodes)
    if not parent_exists:
        drs_nodes.append({
            "id": "",
            "class": "DRSModel",
            "layout": {"x": 0, "y": 0},
            "variables": {}
        })

    for node in nodes:
        node_id = node.get("id")
        if node_id == "root":
            node_id = ""
            
        node_data = node.get("data", {})
        node_class = node_data.get("class", "Module")
        position = node.get("position", {"x": 0, "y": 0})
        variables = node_data.get("variables", {})

        # Find edges
        flow_inputs = []
        data_inputs = []
        variable_reads = []

        for edge in edges:
            if edge.get("target") == node.get("id"):
                target_handle = edge.get("targetHandle")
                source = edge.get("source")
                if source == "root":
                    source = ""
                
                if target_handle == "flow-in":
                    flow_inputs.append(source)
                elif target_handle == "data-in":
                    data_inputs.append(source)
                elif target_handle == "read-in":
                    # Find source node's first variable to read
                    src_node = next((n for n in nodes if n.get("id") == edge.get("source")), None)
                    src_vars = src_node.get("data", {}).get("variables", {}) if src_node else {}
                    var_names = list(src_vars.keys())
                    read_var = var_names[0] if var_names else "value"
                    variable_reads.append({
                        "module": source,
                        "variable": read_var
                    })

        drs_nodes.append({
            "id": node_id,
            "class": node_class,
            "layout": position,
            "variables": variables,
            "connections": {
                "flow_inputs": flow_inputs,
                "data_inputs": data_inputs,
                "variable_reads": variable_reads
            }
        })
    return drs_nodes


class DevServerHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/api/topology":
            file_path = os.path.join(ROOT_DIR, "src", "drs_canvas_state.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    self._set_headers(200)
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                    return
                except Exception as e:
                    self._set_headers(500)
                    self.wfile.write(json.dumps({"error": f"Failed to load file: {e}"}).encode("utf-8"))
                    return
            
            # Fallback to defaults
            self._set_headers(200)
            self.wfile.write(json.dumps({"nodes": DEFAULT_NODES, "edges": DEFAULT_EDGES}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Path '{self.path}' not found"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": f"Invalid JSON body: {e}"}).encode("utf-8"))
            return

        if self.path == "/api/topology":
            canvas_file = os.path.join(ROOT_DIR, "src", "drs_canvas_state.json")
            try:
                os.makedirs(os.path.dirname(canvas_file), exist_ok=True)
                with open(canvas_file, "w") as f:
                    json.dump(payload, f, indent=2)
                
                # Convert & save programmatic flat DRS JSON
                nodes = payload.get("nodes", [])
                edges = payload.get("edges", [])
                drs_flat = react_flow_to_drs_flat(nodes, edges)
                
                with open(os.path.join(WORKSPACE_ROOT, "drs_topology_flat.json"), "w") as f:
                    json.dump(drs_flat, f, indent=2)
                
                # Convert & save hierarchical DRS JSON
                try:
                    from drs.serialize import _flat_canvas_to_tree
                    drs_tree = _flat_canvas_to_tree(drs_flat)
                    with open(os.path.join(WORKSPACE_ROOT, "drs_topology_tree.json"), "w") as f:
                        json.dump(drs_tree, f, indent=2)
                except Exception as tree_ex:
                    print(f"Warning: Hierarchical tree export failed: {tree_ex}")
                
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "ok", "message": "Canvas saved and translated successfully"}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": f"Failed to save topology state: {e}"}).encode("utf-8"))

        elif self.path == "/api/compile":
            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
            drs_flat = react_flow_to_drs_flat(nodes, edges)
            
            try:
                from drs.serialize import compile_canvas_json
                from examples.mining.components.config import ConcentratorConfig
                config = ConcentratorConfig()
                # Compile verification pass
                model = compile_canvas_json(drs_flat, config=config)
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "message": f"Compilation verification successful! Root Class: {type(model).__name__}"
                }).encode("utf-8"))
            except Exception as e:
                err_details = traceback.format_exc()
                self._set_headers(400)
                self.wfile.write(json.dumps({
                    "status": "error",
                    "message": str(e),
                    "details": err_details
                }).encode("utf-8"))
        elif self.path == "/api/simulate":
            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
            max_time = float(payload.get("max_time", 100.0))
            
            drs_flat = react_flow_to_drs_flat(nodes, edges)
            
            try:
                from drs.serialize import compile_canvas_json
                from drs.engine import DRSEngine
                from drs.telemetry import Telemetry
                from examples.mining.components.config import ConcentratorConfig
                
                config = ConcentratorConfig()
                model = compile_canvas_json(drs_flat, config=config)
                engine = DRSEngine(model)
                telemetry = Telemetry(model)
                engine.attach_telemetry(telemetry)
                
                result = engine.run(max_time)
                
                events_out = []
                for e in telemetry.events:
                    events_out.append({
                        "time": e.time,
                        "event_type": e.event_type,
                        "source": e.source,
                        "details": e.details
                    })
                
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "history": telemetry.history,
                    "events": events_out
                }).encode("utf-8"))
            except Exception as e:
                err_details = traceback.format_exc()
                self._set_headers(400)
                self.wfile.write(json.dumps({
                    "status": "error",
                    "message": str(e),
                    "details": err_details
                }).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Path '{self.path}' not found"}).encode("utf-8"))


def run(port=8000):
    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, DevServerHandler)
    print(f"DRS Workspace Local Dev Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dev Server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port=port)
