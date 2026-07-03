import json
from typing import Any
from .module import Module

def save_state(model: Module, filepath: str) -> None:
    """Save a module's state to a JSON file."""
    state = model.state_dict()
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)

def load_state(model: Module, filepath: str) -> None:
    """Load a module's state from a JSON file."""
    with open(filepath, 'r') as f:
        state = json.load(f)
    model.load_state_dict(state)

def export_architecture(model: Module, filepath: str) -> None:
    """Export the module's architecture to a JSON file."""
    arch = model.to_dict()
    with open(filepath, 'w') as f:
        json.dump(arch, f, indent=2)
