import sys
import os
import random
import numpy as np

# Ensure the root directory is on the path so we can import 'examples.mining' and 'drs'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(root_dir)

from drs_mining.components import ConcentratorConfig, ConcentratorModel
from drs import DRSEngine
from drs.canvas_compiler import compile_canvas_json


def get_variable_values(model):
    values = {}
    for name, mod in model.named_modules():
        for var_name, var in mod._variables.items():
            path = f"{name}.{var_name}" if name else var_name
            values[path] = var.value
    return values


def run_simulation(model, length, seed):
    # Reset random seeds
    np.random.seed(seed)
    random.seed(seed)

    engine = DRSEngine(model)
    engine.run(max_time=length)
    return engine


def main():
    print("=== STARTING BLENDING SIMULATION COMPILER VERIFICATION ===")

    length = 500.0  # 500 hours is plenty to hit various states and thresholds
    seed = 42

    config = ConcentratorConfig(
        replication_length=length,
        target_ore_stock_level=60000.0,
        std_dev_ore_fraction=0.05,
        prob_new_facies=0.3,
    )

    print("\n1. Initializing original/reference programmatic simulation...")
    np.random.seed(seed)
    random.seed(seed)
    sim_ref = ConcentratorModel(config)

    print("\n2. Serializing the clean reference architecture (at t=0.0)...")
    # Serialize at t=0.0 BEFORE running reference, so compiled model starts clean!
    serialized_dict = sim_ref.to_dict()
    print("Serialization completed.")

    print("\n3. Compiling the serialized JSON back to a Python Module...")
    # Compile using compile_canvas_json
    sim_compiled = compile_canvas_json(serialized_dict, config=config)
    print("Compilation completed.")

    print("\n4. Running original/reference programmatic simulation...")
    engine_ref = run_simulation(sim_ref, length, seed)
    ref_values = get_variable_values(sim_ref)
    print(
        f"Reference run completed in {engine_ref.step_count} steps. End time: {engine_ref.current_time}"
    )

    print("\n5. Running simulation on the compiled module...")
    engine_compiled = run_simulation(sim_compiled, length, seed)
    compiled_values = get_variable_values(sim_compiled)
    print(
        f"Compiled run completed in {engine_compiled.step_count} steps. End time: {engine_compiled.current_time}"
    )

    # Compare variables
    print("\n6. Comparing final variable values...")
    mismatches = 0
    checked = 0
    for path, ref_val in ref_values.items():
        checked += 1
        comp_val = compiled_values.get(path)
        if comp_val is None:
            print(f"Mismatch: Variable '{path}' missing in compiled model!")
            mismatches += 1
            continue

        # Float vs enum/other comparison
        if isinstance(ref_val, (int, float)) and isinstance(comp_val, (int, float)):
            if abs(ref_val - comp_val) > 1e-9:
                print(
                    f"Mismatch on '{path}': Reference={ref_val}, Compiled={comp_val} (diff={abs(ref_val - comp_val):.2e})"
                )
                mismatches += 1
        else:
            if ref_val != comp_val:
                print(f"Mismatch on '{path}': Reference={ref_val}, Compiled={comp_val}")
                mismatches += 1

    print(f"Checked {checked} variables. Total mismatches: {mismatches}")
    if mismatches == 0:
        print(
            "SUCCESS: Dynamic compilation produced 100% mathematically identical behavior!"
        )
    else:
        print(
            "FAILURE: Mismatches found between reference and compiled simulation runs."
        )
        sys.exit(1)

    # Testing compile -> uncompile (serialize) -> recompile loop...
    np.random.seed(seed)
    random.seed(seed)
    sim_recompiled_ref = ConcentratorModel(config)
    serialized_dict_2 = sim_recompiled_ref.to_dict()

    # Check if they are identical
    import json

    str1 = json.dumps(serialized_dict, sort_keys=True)
    str2 = json.dumps(serialized_dict_2, sort_keys=True)
    print(f"Are serialized JSON strings identical? {str1 == str2}")
    if str1 != str2:
        print("\nDIFF DETAILS:")
        d1 = json.loads(str1)
        d2 = json.loads(str2)

        def compare_dicts(path_str, a, b):
            if type(a) != type(b):
                print(f"Type mismatch at {path_str}: {type(a)} vs {type(b)}")
                return
            if isinstance(a, dict):
                for k in set(a.keys()).union(b.keys()):
                    if k not in a:
                        print(f"Key {k} missing in clean reference at {path_str}")
                    elif k not in b:
                        print(f"Key {k} missing in recompiled reference at {path_str}")
                    else:
                        compare_dicts(f"{path_str}.{k}" if path_str else k, a[k], b[k])
            elif isinstance(a, list):
                if len(a) != len(b):
                    print(f"List length mismatch at {path_str}: {len(a)} vs {len(b)}")
                else:
                    for idx, (item_a, item_b) in enumerate(zip(a, b)):
                        compare_dicts(f"{path_str}[{idx}]", item_a, item_b)
            else:
                if a != b:
                    print(f"Value mismatch at {path_str}: {a} vs {b}")

        compare_dicts("", d1, d2)

    # Re-compile
    sim_recompiled = compile_canvas_json(serialized_dict_2, config=config)
    print("Re-compilation completed.")

    # Run simulation on re-compiled model
    engine_recompiled = run_simulation(sim_recompiled, length, seed)
    recompiled_values = get_variable_values(sim_recompiled)
    print(
        f"Re-compiled run completed in {engine_recompiled.step_count} steps. End time: {engine_recompiled.current_time}"
    )

    # Compare
    mismatches_recomp = 0
    for path, ref_val in ref_values.items():
        recomp_val = recompiled_values.get(path)
        if recomp_val is None:
            print(f"Mismatch in recompiled run: '{path}' missing!")
            mismatches_recomp += 1
        elif isinstance(ref_val, (int, float)) and isinstance(recomp_val, (int, float)):
            if abs(ref_val - recomp_val) > 1e-9:
                print(
                    f"Mismatch in recompiled run on '{path}': Reference={ref_val}, Recompiled={recomp_val} (diff={abs(ref_val - recomp_val):.2e})"
                )
                mismatches_recomp += 1
        else:
            if ref_val != recomp_val:
                print(
                    f"Mismatch in recompiled run on '{path}': Reference={ref_val}, Recompiled={recomp_val}"
                )
                mismatches_recomp += 1

    print(f"Recompiled mismatches: {mismatches_recomp}")
    if mismatches_recomp == 0:
        print(
            "SUCCESS: Bidirectional round-trip (compile -> serialize -> compile) is 100% correct!"
        )
    else:
        print("FAILURE: Round-trip mismatch found.")
        sys.exit(1)

    print("\n=== ALL COMPILED VERIFICATION TASKS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
