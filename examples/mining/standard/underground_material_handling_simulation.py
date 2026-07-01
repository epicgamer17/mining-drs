import os
import sys


sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

if __name__ == "__main__":
    raise SystemExit(
        "Underground material handling sandbox is disabled for now. "
        "Use many_faces_simulation / ActiveFleetConcentratorModel while the model "
        "follows face-level allocation and continuous fleet flow."
    )
