"""Verify canvas-based simulation matches programmatic ConcentratorModel.

Runs both paths with the same config and seeds as the reference
blending_modes/simulation.py script, then compares:

  1. High-level metrics (steps, sim_time, termination reason)
  2. All model variable values (with tolerance)
  3. Full telemetry DataFrames (vectorized per-column comparison)
  4. Mode transition logs from both paths

Usage:
    python scripts/verify_canvas_vs_programmatic.py
"""

import os
import sys
import math
import random
import traceback

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

CANVAS_DIR = os.path.join(WORKSPACE_ROOT, "drs-canvas")
if CANVAS_DIR not in sys.path:
    sys.path.insert(0, CANVAS_DIR)

from drs_dev_server import DEFAULT_NODES, DEFAULT_EDGES, react_flow_to_drs_flat
from drs.serialize import compile_canvas_json
from drs.engine import DRSEngine, SimulationResult
from drs.telemetry import Telemetry
from drs_mining.components import ConcentratorModel, ConcentratorConfig

# Match reference script exactly (blending_modes/simulation.py)
RANDOM_SEED = 11
NUMPY_SEED = 42
MAX_TIME = 99999.0


def seed_all():
    random.seed(RANDOM_SEED)
    np.random.seed(NUMPY_SEED)


def build_config():
    return ConcentratorConfig(
        replication_length=MAX_TIME,
        target_ore_stock_level=60000.0,
        std_dev_ore_fraction=0.05,
        prob_new_facies=0.3,
    )


def collect_vars(mod, prefix=""):
    out = {}
    for name, var in mod._variables.items():
        key = f"{prefix}.{var.name}" if prefix else var.name
        out[key] = var.value
    for child_name, child_mod in mod._modules.items():
        child_key = f"{prefix}.{child_name}" if prefix else child_name
        out.update(collect_vars(child_mod, child_key))
    return out


def print_mode_transition_log(result, label):
    if not result.events:
        print(f"  [{label}] No events logged.")
        return
    state_events = [
        e
        for e in result.events
        if e.event_type == "STATE_CHANGE"
        and e.details.get("variable") == "active_operating_mode"
    ]
    if not state_events:
        print(f"  [{label}] No mode transitions found.")
        return
    print(f"\n  [{label}] Mode Transition Log ({len(state_events)} transitions):")
    for e in state_events:
        old = (
            e.details["old_value"].name
            if hasattr(e.details["old_value"], "name")
            else str(e.details["old_value"])
        )
        new = (
            e.details["new_value"].name
            if hasattr(e.details["new_value"], "name")
            else str(e.details["new_value"])
        )
        print(f"    t={e.time:>10.2f}h  |  {old:>25s}  ->  {new}")


# ---------------------------------------------------------------------------
# Canvas path
# ---------------------------------------------------------------------------
print("=" * 72)
print("Building & running canvas-path simulation ...")
print("=" * 72)

config_canvas = build_config()
seed_all()
drs_flat = react_flow_to_drs_flat(DEFAULT_NODES, DEFAULT_EDGES)
model_canvas = compile_canvas_json(drs_flat, config=config_canvas)

engine_canvas = DRSEngine(model_canvas)
telemetry_canvas = Telemetry(model_canvas)
engine_canvas.attach_telemetry(telemetry_canvas)
result_canvas: SimulationResult = engine_canvas.run(MAX_TIME)
df_canvas = result_canvas.history

print(f"  Steps       : {result_canvas.steps}")
print(f"  Sim time    : {result_canvas.sim_time:.2f}")
print(f"  Termination : {result_canvas.terminated_reason}")
print(f"  History rows: {len(df_canvas) if df_canvas is not None else 0}")

# ---------------------------------------------------------------------------
# Programmatic path
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("Building & running programmatic-path simulation ...")
print("=" * 72)

config_prog = build_config()
seed_all()
model_prog = ConcentratorModel(config_prog, enable_telemetry=False)

engine_prog = DRSEngine(model_prog)
telemetry_prog = Telemetry(model_prog)
engine_prog.attach_telemetry(telemetry_prog)
result_prog: SimulationResult = engine_prog.run(MAX_TIME)
df_prog = result_prog.history

print(f"  Steps       : {result_prog.steps}")
print(f"  Sim time    : {result_prog.sim_time:.2f}")
print(f"  Termination : {result_prog.terminated_reason}")
print(f"  History rows: {len(df_prog) if df_prog is not None else 0}")

# ---------------------------------------------------------------------------
# Compare high-level metrics
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("Comparison")
print("=" * 72)

all_pass = True


def check(label, a, b):
    global all_pass
    ok = a == b
    if not ok:
        print(f"  FAIL {label}: {a!r} != {b!r}")
        all_pass = False
    else:
        print(f"  PASS {label}: {a!r}")


check("Engine steps", result_canvas.steps, result_prog.steps)
check("Simulated time", result_canvas.sim_time, result_prog.sim_time)
check(
    "Termination reason", result_canvas.terminated_reason, result_prog.terminated_reason
)

# ---------------------------------------------------------------------------
# Compare all variable values
# ---------------------------------------------------------------------------
print()
print("--- Model variable values (final) ---")

vars_canvas = collect_vars(model_canvas)
vars_prog = collect_vars(model_prog)

all_var_keys = sorted(set(vars_canvas.keys()) | set(vars_prog.keys()))
var_mismatches = 0
TOL = 1e-5

for key in all_var_keys:
    vc = vars_canvas.get(key, "<missing>")
    vp = vars_prog.get(key, "<missing>")
    if isinstance(vc, float) and isinstance(vp, float):
        if math.isnan(vc) and math.isnan(vp):
            ok = True
        else:
            ok = math.isclose(vc, vp, rel_tol=TOL, abs_tol=TOL)
    elif isinstance(vc, (int, float)) and isinstance(vp, (int, float)):
        ok = abs(vc - vp) < TOL
    else:
        ok = vc == vp
    if not ok:
        print(f"  FAIL var '{key}': canvas={vc!r}  prog={vp!r}")
        var_mismatches += 1
        all_pass = False

if var_mismatches == 0:
    print(f"  PASS all {len(all_var_keys)} variables match")
else:
    print(f"  FAIL {var_mismatches} variable(s) differ (out of {len(all_var_keys)})")

# ---------------------------------------------------------------------------
# Compare telemetry DataFrames (vectorized for performance at ~200k rows)
# ---------------------------------------------------------------------------
print()
print("--- Telemetry DataFrame comparison ---")

if df_canvas is None and df_prog is None:
    print("  PASS both have no telemetry")
elif df_canvas is None or df_prog is None:
    print("  FAIL one has telemetry, other doesn't")
    all_pass = False
else:
    cols_canvas = sorted(df_canvas.columns)
    cols_prog = sorted(df_prog.columns)

    if cols_canvas != cols_prog:
        print("  FAIL column mismatch")
        only_c = set(cols_canvas) - set(cols_prog)
        only_p = set(cols_prog) - set(cols_canvas)
        if only_c:
            print(f"    Only in canvas: {sorted(only_c)}")
        if only_p:
            print(f"    Only in prog:   {sorted(only_p)}")
        all_pass = False
    else:
        df_c = df_canvas[cols_canvas].sort_values("time").reset_index(drop=True)
        df_p = df_prog[cols_prog].sort_values("time").reset_index(drop=True)

        if len(df_c) != len(df_p):
            print(f"  FAIL row count: canvas={len(df_c)} prog={len(df_p)}")
            all_pass = False
        else:
            telemetry_mismatches = 0
            max_mismatches_shown = 10
            # Time column — must be bit-identical
            if not (df_c["time"] == df_p["time"]).all():
                mismatch_idx = (df_c["time"] != df_p["time"]).idxmax()
                print(
                    f"  FAIL 'time' column differs at row {mismatch_idx}: "
                    f"canvas={df_c['time'].iloc[mismatch_idx]} "
                    f"prog={df_p['time'].iloc[mismatch_idx]}"
                )
                all_pass = False
            else:
                # Numeric columns — vectorized allclose per column
                numeric_cols = [
                    c
                    for c in df_c.columns
                    if c != "time" and pd.api.types.is_numeric_dtype(df_c[c])
                ]
                for col in numeric_cols:
                    arr_c = df_c[col].values
                    arr_p = df_p[col].values
                    # Handle NaN/Inf: both arrays must have NaN/Inf at same positions
                    nan_mask_c = np.isnan(arr_c)
                    nan_mask_p = np.isnan(arr_p)
                    if not np.array_equal(nan_mask_c, nan_mask_p):
                        if telemetry_mismatches < max_mismatches_shown:
                            print(f"  FAIL col '{col}': NaN position mismatch")
                        telemetry_mismatches += 1
                        all_pass = False
                        continue
                    inf_mask_c = np.isinf(arr_c)
                    inf_mask_p = np.isinf(arr_p)
                    if not np.array_equal(inf_mask_c, inf_mask_p):
                        if telemetry_mismatches < max_mismatches_shown:
                            print(f"  FAIL col '{col}': Inf position mismatch")
                        telemetry_mismatches += 1
                        all_pass = False
                        continue
                    # Compare only finite values with tolerance
                    finite_mask = ~(nan_mask_c | inf_mask_c)
                    close = (
                        np.allclose(
                            arr_c[finite_mask], arr_p[finite_mask], rtol=TOL, atol=TOL
                        )
                        if finite_mask.any()
                        else True
                    )
                    if not close:
                        abs_diff = np.abs(arr_c - arr_p)
                        worst_idx = np.argmax(abs_diff)
                        if telemetry_mismatches < max_mismatches_shown:
                            print(
                                f"  FAIL col '{col}' at row {worst_idx}: "
                                f"canvas={arr_c[worst_idx]!r}  "
                                f"prog={arr_p[worst_idx]!r}"
                            )
                        telemetry_mismatches += 1
                        all_pass = False

                # Mode column — OperatingMode objects, compare by name
                mode_cols = [
                    c for c in df_c.columns if c not in numeric_cols and c != "time"
                ]
                for col in mode_cols:
                    # Convert to name strings for comparison
                    c_names = df_c[col].apply(
                        lambda x: x.name if hasattr(x, "name") else str(x)
                    )
                    p_names = df_p[col].apply(
                        lambda x: x.name if hasattr(x, "name") else str(x)
                    )
                    if not (c_names == p_names).all():
                        mismatch_idx = (c_names != p_names).idxmax()
                        if telemetry_mismatches < max_mismatches_shown:
                            print(
                                f"  FAIL col '{col}' at row {mismatch_idx}: "
                                f"canvas={c_names.iloc[mismatch_idx]!r}  "
                                f"prog={p_names.iloc[mismatch_idx]!r}"
                            )
                        telemetry_mismatches += 1
                        all_pass = False

            if telemetry_mismatches == 0:
                print(
                    f"  PASS all {len(df_c)} rows x {len(df_c.columns)} columns match"
                )

# ---------------------------------------------------------------------------
# Mode transition logs
# ---------------------------------------------------------------------------
print()
print("--- Mode Transition Logs ---")
print_mode_transition_log(result_canvas, "Canvas")
print_mode_transition_log(result_prog, "Programmatic")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 72)
if all_pass:
    print("RESULT: ALL CHECKS PASSED")
    print(
        "The canvas-based and programmatic simulation paths produce identical results."
    )
else:
    print("RESULT: SOME CHECKS FAILED")
    print("The two simulation paths diverge. See details above.")
print("=" * 72)

sys.exit(0 if all_pass else 1)
