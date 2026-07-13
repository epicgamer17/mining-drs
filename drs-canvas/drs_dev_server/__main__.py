"""Allow running the dev server as `python -m drs_dev_server`."""

import sys
from . import run

port = 8000
if len(sys.argv) > 1:
    try:
        port = int(sys.argv[1])
    except ValueError:
        pass

run(port=port)
