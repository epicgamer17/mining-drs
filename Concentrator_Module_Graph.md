# DRS Module Graph — ConcentratorModel

> Generated automatically by `drs.vis.module_graph.save_module_graph_report`

## Module Hierarchy

| Name | Path | Variables |
|------|------|-----------|
| `ConcentratorModel` | `(root)` | `global_time` |
| `mine` | `mine` | `active_parcel_initial_mass, cumulative_extracted_mass, parcel_extracted_mass, active_parcel_ore_fraction` |
| `generator` | `mine.generator` | `—` |
| `fleet` | `fleet` | `stockpile2_routing_fraction` |
| `ore1_stock` | `ore1_stock` | `current_mass, actual_outflow_rate, contained_ore_fraction_mass` |
| `ore2_stock` | `ore2_stock` | `current_mass, actual_outflow_rate, contained_ore_fraction_mass` |
| `plant` | `plant` | `cumulative_milled_mass` |
| `controller` | `controller` | `active_operating_mode, total_system_ore_mass, current_campaign_duration, current_contingency_duration, cumulative_time_mode_a, +9 more` |

## Flowchart

```mermaid
flowchart TD
subgraph root["<b>ConcentratorModel</b>"]
    root_vars[/"<b>ConcentratorModel</b> vars<br><i>global_time</i>"\]
    style root_vars fill:transparent,stroke-dasharray: 5 5
    subgraph mine["<b>mine</b>"]
        mine_vars[/"<b>mine</b> vars<br><i>active_parcel_initial_mass</i><br><i>cumulative_extracted_mass</i><br><i>parcel_extracted_mass</i><br><i>active_parcel_ore_fraction</i>"\]
        style mine_vars fill:transparent,stroke-dasharray: 5 5
        mine_generator(["<b>generator</b>"])
    end
    fleet(["<b>fleet</b><br><i>stockpile2_routing_fraction</i>"])
    ore1_stock(["<b>ore1_stock</b><br><i>current_mass</i><br><i>actual_outflow_rate</i><br><i>contained_ore_fraction_mass</i>"])
    ore2_stock(["<b>ore2_stock</b><br><i>current_mass</i><br><i>actual_outflow_rate</i><br><i>contained_ore_fraction_mass</i>"])
    plant(["<b>plant</b><br><i>cumulative_milled_mass</i>"])
    controller(["<b>controller</b><br><i>active_operating_mode</i><br><i>total_system_ore_mass</i><br><i>current_campaign_duration</i><br><i>current_contingency_duration</i><br><i>cumulative_time_mode_a</i><br><i>+9 more</i>"])
end
    controller -->|target_mine_mass_rate| mine
    controller -->|target_stock1_outflow_rate| ore1_stock
    controller -->|target_stock2_outflow_rate| ore2_stock
    mine -->|cumulative_extracted_mass| controller
    ore1_stock -->|Ore1Stock_mass| controller
    ore2_stock -->|Ore2Stock_mass| controller
    fleet -->|stockpile2_routing_fraction| controller
    mine ==>|flow| fleet
    fleet ==>|flow| ore1_stock
    fleet ==>|flow| ore2_stock
    ore1_stock ==>|flow| plant
    ore2_stock ==>|flow| plant
```

## Data Dependencies (persistent variable reads)

The following read-dependencies were recorded during the simulation. An arrow `A → B` means module B reads a variable owned by module A.

  - `ore1_stock` → `ConcentratorModel` reads `Ore1Stock_mass`
  - `ore2_stock` → `ConcentratorModel` reads `Ore2Stock_mass`
  - `controller` → `ConcentratorModel` reads `total_system_ore_mass`
  - `controller` → `mine` reads `target_mine_mass_rate`
  - `controller` → `ore1_stock` reads `target_stock1_outflow_rate`
  - `controller` → `ore2_stock` reads `target_stock2_outflow_rate`
  - `mine` → `controller` reads `cumulative_extracted_mass`
  - `ore1_stock` → `controller` reads `Ore1Stock_mass`
  - `ore2_stock` → `controller` reads `Ore2Stock_mass`
  - `fleet` → `controller` reads `stockpile2_routing_fraction`

## Data Flow (transient)

The following transient flow-edges were recorded during the simulation. An arrow `A → B` means module A returned a `drs.Flow` value that was passed as input to module B.

  - `mine` → `fleet` flow
  - `fleet` → `ore1_stock` flow
  - `fleet` → `ore2_stock` flow
  - `ore1_stock` → `plant` flow
  - `ore2_stock` → `plant` flow

## Visual Graph

![Module Graph](Concentrator_Module_Graph.png)
