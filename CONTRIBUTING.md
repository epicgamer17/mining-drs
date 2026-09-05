# Contributing to mining-drs

The goal of this repository is to provide a clean, high-performance, and mathematically rigorous mining library that can be used for any mining engineering, computer science, and mine optimization research.

---

## Part 1: General Design & Engineering Principles

### 1. Python First & Simple Over Easy
- **Functional Over OOP**: Prefer clean, composable functions over deep class inheritance hierarchies. Classes should be reserved for stateful physical assets (like stockpiles, processors, or gym environments).
- **Minimal Indirection**: Avoid abstraction layers for their own sake. Functions should accept clean arrays or DataFrames and return standard arrays or DataFrames.
- **No Backwards Compatibility Wrappers**: Clean APIs take precedence over legacy backwards compatibility. When an API improves, update existing callers and tests directly rather than carrying deprecated shims or fallbacks.
- **Adhere to python-drs**: Avoid building custom internal components when existing `python-drs` primitives and objects can be utilized. Components should be generic and hardcode as little domain-specific logic as possible.

### 2. The "No Non-Mine-Agnostic Defaults" Philosophy
> **Rule**: If a parameter varies by commodity, deposit geology, rock type, or equipment fleet, it **must not** have a default value. It must be a required argument.

Providing hardcoded defaults for deposit-specific or site-specific parameters creates hazardous silent assumptions that produce invalid mine models or misleading research results. Forcing callers to pass these parameters ensures they are deliberately chosen:

- **Strictly Prohibited Defaults (Must Be Explicitly Supplied)**:
  - **Physical Rock Properties**: Bulk density / specific gravity, moisture content.
  - **Haulage & Equipment Fleet**: Fixed cycle time, haul/return speeds, payload capacity, spotting/dump times.
  - **Economic & Valuation Parameters**: Commodity prices, mining costs, milling/processing costs, G&A costs, refining deductions (TC/RCs), royalties, smelter payability.
  - **Resource Classification Parameters**: Drillhole search radii for Measured/Indicated, minimum informing hole counts, Slope of Regression thresholds, Kriging Efficiency thresholds.
  - **Modifying Factors**: Mining dilution %, mining recovery %, metallurgical recovery %, economic cut-off grades.
- **Acceptable Defaults**:
  - Pure mathematical/algorithmic constants where an established standard exists (e.g., IDW exponent `power=2.0`, metal factor conversion `0.01` for %, significant figures `sig_figs=3`, stochastic simulation count `n_realizations=50`).
  - Standard column name strings (e.g., `grade_col="grade"`, `x_col="x"`, `tonnes_col="tonnes"`).
  - Cosmetic visualization parameters (e.g., figure sizes, titles, colormaps).

### 3. Transparent Data Structures & Config Objects
- **DataFrames and NumPy Arrays Only**: Inputs and outputs should remain standard pandas DataFrames or numpy ndarrays.
- **Avoid Dataclass Config Objects**: Inline function hyperparameters and simulation parameters directly. Dataclass configs create unnecessary boilerplate and increase the risk of silent site-specific defaults leaking into the core library.
- **Metadata via `df.attrs`**: Compliance footnotes, reconciliation health status, audit summaries, and conversion factors should be attached to DataFrame metadata (`df.attrs["footnotes"]`), keeping DataFrame columns clean for vector operations.

### 4. Strict Input Validation at Function Boundaries
Functions must validate input arguments immediately upon entry and fail fast with informative errors:
- Physical quantities must be positive (`bulk_density > 0`, `fixed_cycle_time_min > 0`, `block_size > 0`).
- Classification radii must be ordered: `max_radius_measured <= max_radius_indicated`.
- Thresholds must be ordered: `threshold_measured > threshold_indicated`.
- Raise `TypeError` for missing required parameters and descriptive `ValueError` for invalid parameter values or mismatched array dimensions.

### 5. Visualization Suite Standards
- **Identical Scale Norms**: When plotting block models against informing drillholes, always use identical colormaps and normalized scale bins (`BoundaryNorm` or `Normalize`) to prevent visual scale distortion.
- **Dual-Axis Presentation**: Grade–Tonnage curves must present Ore Tonnage on the primary left axis (decreasing monotonically) and Average Ore Grade on the secondary right axis (increasing monotonically).
- **Multi-View Block Model Suites**: Provide both 2D orthogonal slices (bench plan, cross section, longitudinal section) and 3D isometric/interactive projections for 3D spatial methods.

---

## Part 2: Mining Standards, Geostatistics & Regulatory Compliance

As much as possible, implementations must adhere to the **SME Mining Engineering Handbook (3rd Ed.)** and international statutory reporting standards (**CIM**, **JORC**, **CRIRSCO**, **SEC Regulation S-K 1300**). Authoritative reference PDFs are maintained in `mining_standards/`.

### 1. Statutory & Regulatory Citation Conventions
- **Descriptive Citations**: Rather than citing fragile sub-clause numbers that shift across standard revisions (e.g., avoid nonexistent CIM MRMR `§7.16`), cite the authoritative body, year, and descriptive topic/section:
  - *CIM Definition Standards (2014)*
  - *CIM MRMR Best Practice Guidelines (2019, Section 7 on Mineral Reserve Estimation and Internal Waste)*
  - *CIM Exploration Best Practice Guidelines (2018, §4.3 & §5.1 on Core Recovery and Gap Treatment)*
  - *JORC Code (2012, Clause 29 on Modifying Factors and Ore Reserves)*
  - *SEC Regulation S-K 1300 (§229.1303(b)(1) & (b)(2) on Point of Reference and Resource Exclusivity)*
  - *CRIRSCO International Reporting Template (Clause 28 & Clause 38)*
- **Mandatory Verification Tag**: Every external regulatory reference in code docstrings, comments, or report footnotes must include the tag:
  ```python
  (TODO: Manually Verify)
  ```
  This allows automated grepping (`grep "(TODO: Manually Verify)"`) to cross-reference citations directly against authoritative texts in `mining_standards/`.

### 2. Spatial Interpolation vs. Extrapolation Control
Every spatial estimation method (Polygonal, Nearest Neighbor, Inverse Distance Weighting, Kriging) must strictly differentiate between informing data support and un-drilled space:
- **Convex Hull Confinement**: Provide an option to clip or mask estimates strictly to the sample convex hull (e.g., `clip_to_convex_hull=True` or `mask_extrapolation=True`).
- **Maximum Search Radius**: Provide `max_radius` to restrict spatial extrapolation around informing points.
- **Boundary vs. Data Support Separation**: Distinctly separate the concession / pit perimeter (`boundary`) from the actual drillhole support envelope (`is_within_convex_hull`). Never assume the legal perimeter equals geological data support.

### 3. Geological Domain Segregation & Compositing Standards
- **Hard Domain Contacts**: Spatial estimation (IDW, Kriging, Polygonal), variography, and down-hole compositing must strictly respect geological and lithological domains (`domain_col`). Estimators must never share samples across hard geological boundaries.
- **Compositing Contact Resets**: Compositing must reset at domain boundaries. Non-contiguous intervals of the same lithology (e.g., $A \to B \to A$) must be treated as independent runs so boundary contamination is avoided.
- **Unassayed Core Gaps**: Never assume unsampled core loss carries the grade of adjacent mineralization without explicit justification (CIM Exploration Guidelines §4.3 & JORC Table 1). Down-hole compositors must provide explicit treatments: conservative zero-grade insertion (`unassayed_treatment="zero"`), gap splitting (`unassayed_treatment="split"`), or strict exception raising (`"error"`).

### 4. Multi-Criteria Resource Classification & The Conservative Downgrade Principle
- **No Standalone Geometric Metric**: Never classify mineral resources based solely on Kriging estimation variance or drill spacing alone (CIM MRMR §6.11 explicitly cautions against this due to geometric bias and "spotted dog" artifacts).
- **Multi-Criteria Integration**: Combine drillhole spacing, informing hole counts, estimation variance, and Kriging Neighborhood Analysis Slope of Regression / Kriging Efficiency.
- **The Conservative Downgrade Principle**:
  $$\text{Category} = \min(\text{Spacing}, \text{Variance}, \text{SoR})$$
  A block only attains high confidence (e.g., Measured) if it satisfies all active criteria. Failure of any single criterion downgrades the block to the lowest satisfied category.
- **Spatial Smoothing**: Automated numerical classifications must provide spatial post-processing smoothing (majority/mode filter or minimum cluster sizing via `smooth_resource_categories`) to eliminate disjointed single-block artifacts before mine planning.

### 5. Modifying Factors & Mineral Reserve Delineation
- **Strict Inferred Exclusion**: Inferred Mineral Resources can **never** be converted into Mineral Reserves under any circumstances due to geological uncertainty (CRIRSCO, CIM, JORC, SEC S-K 1300).
- **Classification Mapping**: Measured Resources convert to Proven (or Proved) Reserves; Indicated Resources convert to Probable Reserves.
- **Cut-Off Grade Support**:
  - Differentiate between **Run-of-Mine (ROM) cut-off** (tested against diluted delivered head grade at the plant gate) and **in-situ cut-off** (dilution-adjusted).
  - Reserve cut-off grades must include sustaining capital (tailings expansions, replacement equipment per CIM Table 7-1), whereas Resource RPEEE cut-off grades reflect operating cash costs.
- **Mandatory Statutory Disclosures**: Reserve and resource reporting tables must disclose:
  1. Point of Reference (e.g., `"In situ"`, `"Run-of-Mine delivered to plant"`, or `"Marketable product"`).
  2. Inclusive vs. exclusive resource declaration relative to reserves.
  3. Applied Modifying Factors (mining dilution %, mining recovery %, metallurgical recovery %, commodity price).

### 6. Production Reconciliation & Mass Conservation
- **Harry Parker (2012) Framework**: Implements the standard $F_1$ (Model to Mine), $F_2$ (Mine to Mill), and $F_3$ (Total Value Chain) factors.
- **Stockpile Accounting**: When intermediate stockpiling occurs, mass conservation of inventory movements (adds/reclaims or balance deltas) must be applied:
  $$\Delta M_{\text{stockpile}} = M_{\text{close}} - M_{\text{open}} = M_{\text{added}} - M_{\text{reclaimed}}$$
  $$M_{\text{plant, adj}} = M_{\text{plant}} + \Delta M_{\text{stockpile}}$$
  $$F_{2,\text{adj}} = \frac{M_{\text{plant, adj}}}{M_{\text{grade\_control}}}, \quad F_{3,\text{adj}} = \frac{M_{\text{plant, adj}}}{M_{\text{reserve}}}$$
  Failure to account for stockpiling produces false-alarm drops during stockpiling (apparent ore loss) and false-alarm surges during reclaim (phantom metal creation).
- **Mathematical Reconciliation Identities**:
  - $F_1 \times F_{2,\text{adj}} = F_{3,\text{adj}}$
  - $R_{\text{tonnes}} \times R_{\text{grade}} = F_{\text{metal}}$