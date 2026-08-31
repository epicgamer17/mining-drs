"""Dedicated Fleet Operating Mode Controller for Underground Mine Haulage.

Implements two discrete fleet operating modes:
1. PRODUCTION:
   - Ore haulage is primary to supply the mill.
   - Extra / surplus trucks prioritize Sustaining Stope Turnaround Development.
   - If no stope is in turnaround, extra trucks fallback to Capital Decline Development.

2. DEVELOPMENT:
   - Ore haulage is still primary to prevent plant starvation.
   - Extra / surplus trucks prioritize Area 2 Capital Decline Development.
   - If capital decline is completed or heading unavailable, extra trucks fallback to Stope Turnaround Development.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from .fleet import MissionType, Truck, TruckPhase
from .mine_face import MineFace, FaceState


class FleetOperatingMode(Enum):
    """Discrete fleet operating mode."""

    PRODUCTION = "PRODUCTION"
    DEVELOPMENT = "DEVELOPMENT"


class FleetModeController:
    """Supervisory controller managing fleet operating mode transitions and truck mission selection."""

    def __init__(
        self,
        initial_mode: FleetOperatingMode = FleetOperatingMode.PRODUCTION,
        target_stockpile_buffer_tonnes: float = 60000.0,
        min_production_trucks: int = 4,
        reserved_dev_trucks: int = 2,
    ):
        self.current_mode: FleetOperatingMode = initial_mode
        self.target_stockpile_buffer_tonnes: float = target_stockpile_buffer_tonnes
        self.min_production_trucks: int = min_production_trucks
        self.reserved_dev_trucks: int = reserved_dev_trucks
        self.mode_history: List[Tuple[float, FleetOperatingMode]] = [(0.0, initial_mode)]

    @property
    def mode(self) -> FleetOperatingMode:
        """Returns the currently active fleet operating mode."""
        return self.current_mode

    def evaluate_mode(
        self,
        policy: int,
        current_day: float,
        dev_progress_m: float,
        required_dev_m: float,
        deadline_day: Optional[float],
        area2_locked: bool,
        plant_operating_mode: str = "MODE_A",
    ) -> FleetOperatingMode:
        """Evaluates and updates the active fleet operating mode.

        - Policy 1 (Myopic Baseline): Strictly operates in PRODUCTION mode during active campaigns.
        - Policy 2 (Hierarchical Control): Coupled to Mill Operating Mode:
          - In MODE_A (and submodes): Operates in PRODUCTION mode.
          - In MODE_B: Operates in DEVELOPMENT mode while Area 2 is locked.
          - Emergency overrides: In MODE_B_CONTINGENCY or MODE_B_MINE_SURGING, operates in PRODUCTION
            to protect plant feed and recover depleted stockpiles.
          - Once Area 2 is unlocked or development is completed, operates in PRODUCTION mode.
        """
        if policy == 1 or not area2_locked or required_dev_m <= 0.0:
            new_mode = FleetOperatingMode.PRODUCTION
        else:
            mode_upper = str(plant_operating_mode).upper()
            if mode_upper.startswith("MODE_B") and not (
                "_CONTINGENCY" in mode_upper or "_MINE_SURGING" in mode_upper
            ):
                new_mode = FleetOperatingMode.DEVELOPMENT
            else:
                new_mode = FleetOperatingMode.PRODUCTION

        if new_mode != self.current_mode:
            self.current_mode = new_mode
            self.mode_history.append((current_day, new_mode))

        return self.current_mode

    def select_mission(
        self,
        truck: Truck,
        current_total_stock: float,
        is_plant_shutdown: bool,
        can_mine_ore: bool,
        active_prod_trucks: int,
        active_capital_dev_trucks: int,
        faces: Sequence[MineFace],
        area2_locked: bool,
        preferred_face_id: int,
        face_levels: Mapping[int, int],
        default_area1_level: int = 3,
        default_area2_level: int = 6,
        daily_target_ore: float = 6000.0,
        daily_hauled_ore: float = 0.0,
        is_surging: bool = False,
        area1_exhausted: bool = False,
        reserved_dev_trucks: Optional[int] = None,
    ) -> Optional[Tuple[MissionType, int, int, bool]]:
        """Determines the exact mission assignment for an available haul truck.

        Returns:
            Tuple of (mission_type, target_face_id, target_level, is_waste), or None if truck should remain idle.
        """
        # Scheduled site-wide maintenance shutdown (mill and fleet maintenance): trucks stand down
        if is_plant_shutdown:
            return None

        # 60kt semi-hard stockpile cap: halt ore haulage when full unless explicitly surging
        stock_full = (
            current_total_stock >= (self.target_stockpile_buffer_tonnes - 1e-6)
        ) and not is_surging

        # Find active stope currently in turnaround
        stope_in_turnaround = next(
            (f for f in faces if f.state == FaceState.DEVELOPMENT_TURNAROUND),
            None,
        )

        # -------------------------------------------------------------------
        # Mode-Specific Mission Selection
        # -------------------------------------------------------------------
        if self.current_mode == FleetOperatingMode.DEVELOPMENT:
            n_res = (
                reserved_dev_trucks
                if reserved_dev_trucks is not None
                else self.reserved_dev_trucks
            )

            # 1. Strategic Capital Decline Development Push (up to reserved dev trucks)
            if area2_locked and active_capital_dev_trucks < n_res:
                return MissionType.CAPITAL_DECLINE_DEV, -1, default_area2_level, True

            # 2. Stope Turnaround Development
            if stope_in_turnaround is not None:
                lvl = face_levels.get(stope_in_turnaround.face_id, default_area1_level)
                return (
                    MissionType.STOPE_TURNAROUND_DEV,
                    stope_in_turnaround.face_id,
                    lvl,
                    True,
                )

            # 3. Sustaining Ore Production (if stock < 60kt)
            if not stock_full and can_mine_ore:
                target_face_id = preferred_face_id
                target_face = next(
                    (f for f in faces if f.face_id == target_face_id), None
                )
                if target_face is not None and target_face.is_ore_available:
                    target_level = face_levels.get(
                        target_face_id,
                        default_area2_level
                        if target_face_id == 2
                        else default_area1_level,
                    )
                    return MissionType.ORE_HAUL, target_face_id, target_level, False

            # 4. Surplus Capacity to Capital Decline (if Area 2 locked)
            if area2_locked:
                return MissionType.CAPITAL_DECLINE_DEV, -1, default_area2_level, True

            return None

        else:
            # ---------------------------------------------------------------
            # PRODUCTION Mode (Policy 1, and Policy 2 in Mode A):
            # ---------------------------------------------------------------
            # 1. Stope Turnaround Development First (muck waste to blast next round)
            if stope_in_turnaround is not None:
                lvl = face_levels.get(stope_in_turnaround.face_id, default_area1_level)
                return (
                    MissionType.STOPE_TURNAROUND_DEV,
                    stope_in_turnaround.face_id,
                    lvl,
                    True,
                )

            # 2. Ore Haulage from Active Stopes (while stock < 60kt)
            if not stock_full and can_mine_ore:
                target_face_id = preferred_face_id
                target_face = next(
                    (f for f in faces if f.face_id == target_face_id), None
                )
                if target_face is not None and target_face.is_ore_available:
                    target_level = face_levels.get(
                        target_face_id,
                        default_area2_level
                        if target_face_id == 2
                        else default_area1_level,
                    )
                    return MissionType.ORE_HAUL, target_face_id, target_level, False

            # 3. Strict Area 1 Containment: While Area 1 is active, NEVER leak to Area 2!
            if not area1_exhausted:
                return None

            # 4. Post-Depletion: Only when Area 1 is 100% exhausted does Policy 1 develop Area 2
            if area2_locked:
                return MissionType.CAPITAL_DECLINE_DEV, -1, default_area2_level, True

            return None
