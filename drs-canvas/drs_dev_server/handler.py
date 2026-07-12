import http.server
import json
import os
import traceback

from .defaults import DEFAULT_NODES, DEFAULT_EDGES
from .converters import react_flow_to_drs_flat, _remap_telemetry_keys
from .dashboard import generate_dashboard_plot

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(ROOT_DIR)
CANVAS_STATE_FILE = os.path.join(ROOT_DIR, "src", "drs_canvas_state.json")
FLAT_EXPORT_FILE = os.path.join(WORKSPACE_ROOT, "drs_topology_flat.json")
TREE_EXPORT_FILE = os.path.join(WORKSPACE_ROOT, "drs_topology_tree.json")


class DevServerHandler(http.server.BaseHTTPRequestHandler):
    def _respond(self, status=200, body=None, content_type="application/json"):
        self.send_response(status)
        body_bytes = body.encode("utf-8") if isinstance(body, str) else body
        self.send_header("Content-Type", content_type)
        if body_bytes is not None:
            self.send_header("Content-Length", str(len(body_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body_bytes is not None:
            self.wfile.write(body_bytes)

    def _json_response(self, status, data):
        from . import _serialize_for_response

        self._respond(
            status, body=json.dumps(_serialize_for_response(data)).encode("utf-8")
        )

    def _error_response(self, status, message, details=None):
        self._json_response(status, {"error": message, "details": details})

    def do_OPTIONS(self):
        self._respond(200)

    def do_GET(self):
        if self.path == "/api/topology":
            self._handle_get_topology()
        else:
            self._error_response(404, f"Path '{self.path}' not found")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self._error_response(400, f"Invalid JSON body: {e}")
            return

        if self.path == "/api/topology":
            self._handle_post_topology(payload)
        elif self.path == "/api/compile":
            self._handle_post_compile(payload)
        elif self.path == "/api/simulate":
            self._handle_post_simulate(payload)
        else:
            self._error_response(404, f"Path '{self.path}' not found")

    def _handle_get_topology(self):
        if os.path.exists(CANVAS_STATE_FILE):
            try:
                with open(CANVAS_STATE_FILE, "r") as f:
                    data = json.load(f)
                self._json_response(200, data)
                return
            except Exception as e:
                self._error_response(500, f"Failed to load file: {e}")
                return
        self._json_response(200, {"nodes": DEFAULT_NODES, "edges": DEFAULT_EDGES})

    def _handle_post_topology(self, payload):
        try:
            os.makedirs(os.path.dirname(CANVAS_STATE_FILE), exist_ok=True)
            with open(CANVAS_STATE_FILE, "w") as f:
                json.dump(payload, f, indent=2)
            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
            drs_flat = react_flow_to_drs_flat(nodes, edges)
            with open(FLAT_EXPORT_FILE, "w") as f:
                json.dump(drs_flat, f, indent=2)
            try:
                from drs.canvas_compiler import _flat_canvas_to_tree

                drs_tree = _flat_canvas_to_tree(drs_flat)
                with open(TREE_EXPORT_FILE, "w") as f:
                    json.dump(drs_tree, f, indent=2)
            except Exception as tree_ex:
                print(f"Warning: Hierarchical tree export failed: {tree_ex}")
            self._json_response(
                200,
                {"status": "ok", "message": "Canvas saved and translated successfully"},
            )
        except Exception as e:
            self._error_response(500, f"Failed to save topology state: {e}")

    def _handle_post_compile(self, payload):
        try:
            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
            drs_flat = react_flow_to_drs_flat(nodes, edges)
            from drs.canvas_compiler import compile_canvas_json
            from drs_mining.components.config import ConcentratorConfig

            config = ConcentratorConfig()
            model = compile_canvas_json(drs_flat, config=config)
            self._json_response(
                200,
                {
                    "status": "ok",
                    "message": f"Compilation verification successful! Root Class: {type(model).__name__}",
                },
            )
        except Exception as e:
            self._error_response(400, str(e), traceback.format_exc())

    def _handle_post_simulate(self, payload):
        try:
            import random as _random
            import numpy as _numpy

            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
            max_time = float(payload.get("max_time", 100.0))
            seed_value = int(payload.get("seed", 42))

            _random.seed(seed_value)
            _numpy.random.seed(seed_value)

            drs_flat = react_flow_to_drs_flat(nodes, edges)

            from drs.canvas_compiler import compile_canvas_json
            from drs.engine import DRSEngine
            from drs.telemetry import Telemetry
            from drs_mining.components.config import ConcentratorConfig

            config = ConcentratorConfig()
            model = compile_canvas_json(drs_flat, config=config)
            engine = DRSEngine(model)
            telemetry = Telemetry(model)

            telemetry.register_metric(
                "MassOfCurrentParcel",
                lambda t, m, s, _: m.mine.active_parcel_initial_mass.value,
            )
            telemetry.register_metric(
                "CurrentParcelRoutingFraction",
                lambda t, m, s, _: m.fleet.stockpile2_routing_fraction.value,
            )
            telemetry.register_metric(
                "Campaign_Shutdown",
                lambda t, m, s, _: m.controller.current_campaign_duration.value,
            )
            telemetry.register_metric(
                "Contingency",
                lambda t, m, s, _: m.controller.current_contingency_duration.value,
            )

            engine.attach_telemetry(telemetry)
            result = engine.run(max_time)

            remapped_history = _remap_telemetry_keys(telemetry.history, model)

            import pandas as _pd

            _df = _pd.DataFrame(telemetry.history)
            _dashboard_png = generate_dashboard_plot(_df)

            events_out = [
                {
                    "time": e.time,
                    "event_type": e.event_type,
                    "source": e.source,
                    "details": e.details,
                }
                for e in telemetry.events
            ]

            self._json_response(
                200,
                {
                    "status": "ok",
                    "history": remapped_history,
                    "events": events_out,
                    "plots": {"dashboard_png": _dashboard_png},
                },
            )
        except Exception as e:
            self._error_response(400, str(e), traceback.format_exc())
