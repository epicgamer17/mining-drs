"""
Differences between this DRS Implementation and the 2017 Shelswell Paper:

1. Simulation Methodology (Discrete Rate vs. Discrete Event):
   - Paper: A true Discrete Event Simulation (DES) where individual trucks and loader units are simulated as independent entities executing discrete tasks (traveling, spotting, loading, queuing, dumping, refuelling, breaking down).
   - Implementation: A Discrete Rate Simulation (DRS) where flow capacities are represented as continuous rates (tonnes/day). Capacity limits are solved analytically at each time step rather than simulated step-by-step.

2. Truck & Loader Cycles:
   - Paper: Truck loading times are simulated using a series of stochastic LHD loading bucket cycles, spot times, and uniform acquisition delays.
   - Implementation: The loading cycle is represented analytically based on average load spot, average acquisition delay, and LHD bucket cycle times (empty/loaded LHD tram times + bucket spot + load + dump times).

3. Operator Pooling and Fleet Constraints:
   - Paper: Explicitly models operators and trucks as individual resources that must be acquired. Available trucks remain idle if operators are unavailable.
   - Implementation: Represented analytically by capping the effective fleet size: eff_trucks = min(N_trucks * Availability, N_operators).

4. Mechanical Availability:
   - Paper: Simulates scheduled planned maintenance (PM) at utilization intervals and random breakdowns using MTBF/MTTR probability distributions.
   - Implementation: Approximated analytically by scaling down the effective fleet size by the overall mechanical availability bracket.

5. Traffic & Congestion Delays:
   - Paper: Restricts decline/ramp roadway segments to single-direction travel with passing pull-outs. Loaded trucks are prioritized, and congestion creates queuing bottlenecks.
   - Implementation: Approximated using a linear traffic delay penalty applied to truck cycle times, scaled by the number of trucks allocated per level.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from drs import DRSEngine
from drs_mining.components.config import ShelswellConfig
from drs_mining.components.shelswell_controller import ShelswellHaulageModel

def run_simulation(trucks, operators, availability):
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    config = ShelswellConfig()
    
    # Run parameters
    model = ShelswellHaulageModel(config)
    model.truck_count.value = float(trucks)
    model.operator_count.value = float(operators)
    model.availability.value = float(availability)
    
    engine = DRSEngine(model)
    # Run for 354 production days
    engine.run(max_time=354.0)
    
    total_hauled = model.ore_hauled.value + model.waste_hauled.value
    avg_daily_productivity = total_hauled / 354.0
    return avg_daily_productivity

def generate_figure_2():
    print("Generating Figure 2 (Productivity vs Fleet Size without operator constraints)...")
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    truck_sizes = list(range(3, 11))
    
    # Base normalization value: 10 trucks, 10 operators at 1.0 availability
    base_prod = run_simulation(10, 10, 1.0)
    print(f"Base productivity (10 trucks, 10 ops, 100% avail): {base_prod:.2f} t/d")
    
    plt.figure(figsize=(10, 6))
    
    colors = {
        0.5: "green",
        0.6: "brown",
        0.7: "orange",
        0.8: "blue",
        0.9: "hotpink",
        1.0: "black"
    }
    
    for avail in availabilities:
        prod_list = []
        for trucks in truck_sizes:
            prod = run_simulation(trucks, trucks, avail)
            prod_list.append(prod / base_prod)
        
        plt.plot(truck_sizes, prod_list, marker='o', label=f"{int(avail*100)}% availability", color=colors[avail])
        
    plt.title("Haulage productivity analysis without haulage operator constraints", fontsize=14)
    plt.xlabel("Number of trucks in fleet", fontsize=12)
    plt.ylabel("Normalised productivity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.ylim(0, 1.1)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/shelswell_fig2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/shelswell_fig2.png")

def generate_figures_3_to_8():
    availabilities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    operator_counts = list(range(1, 11))
    truck_counts = list(range(3, 11))
    
    colors = {
        3: "green",
        4: "brown",
        5: "orange",
        6: "blue",
        7: "hotpink",
        8: "purple",
        9: "yellow",
        10: "black"
    }
    
    for avail in availabilities:
        print(f"Generating productivity curves for {int(avail*100)}% mechanical availability...")
        # Base normalization: 10 trucks, 10 operators at the current availability
        base_prod = run_simulation(10, 10, avail)
        
        plt.figure(figsize=(10, 6))
        for trucks in truck_counts:
            prod_list = []
            for ops in operator_counts:
                # Operator count is capped at number of trucks for this sweep
                prod = run_simulation(trucks, min(ops, trucks), avail)
                prod_list.append(prod / base_prod)
            
            plt.plot(operator_counts, prod_list, marker='o', label=f"{trucks} trucks", color=colors[trucks])
            
        plt.title(f"Haulage fleet size productivity analysis with operator constraints ({int(avail*100)}% Availability)", fontsize=13)
        plt.xlabel("Number of haulage operators", fontsize=12)
        plt.ylabel("Normalised productivity", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="lower right")
        plt.ylim(0, 1.1)
        
        fig_num = {1.0: 3, 0.9: 4, 0.8: 5, 0.7: 6, 0.6: 7, 0.5: 8}[avail]
        plt.savefig(f"plots/shelswell_fig{fig_num}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plots/shelswell_fig{fig_num}.png")

if __name__ == "__main__":
    generate_figure_2()
    generate_figures_3_to_8()
    print("Replication of all figures complete!")
