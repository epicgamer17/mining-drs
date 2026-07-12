import http.server
import math
import os
import sys

from drs.variables import serialize_val

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(ROOT_DIR)
sys.path.append(WORKSPACE_ROOT)


def _serialize_for_response(obj):
    # Delegates base scalar serialization to drs.variables.serialize_val
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_for_response(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_response(v) for v in obj]
    return serialize_val(obj)


from .defaults import DEFAULT_NODES, DEFAULT_EDGES
from .converters import react_flow_to_drs_flat
from .handler import DevServerHandler


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
