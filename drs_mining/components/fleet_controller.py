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
    ):
        self.current_mode: FleetOperatingMode = initial_mode
        self.target_stockpile_buffer_tonnes: float = target_stockpile_buffer_tonnes
        self.min_production_trucks: int = min_production_trucks
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
    ) -> FleetOperatingMode:
        """Evaluates and updates the active fleet operating mode.

        - Policy 1 (Myopic Baseline): Strictly operates in PRODUCTION mode.
        - Policy 2 (Hierarchical Control): Evaluates trajectory ratio R(t) = dev(t) / expected_dev(t).
          When lagging behind schedule (R(t) < 1.0) and Area 2 is locked, switches to DEVELOPMENT mode.
          When on track (R(t) >= 1.0) or once Area 2 is unlocked, operates in PRODUCTION mode.
        """
        if policy == 1 or not area2_locked or required_dev_m <= 0.0:
            new_mode = FleetOperatingMode.PRODUCTION
        else:
            # Policy 2 trajectory evaluation
            if deadline_day is not None and deadline_day > 0:
                elapsed_frac = max(1e-4, min(1.0, current_day / deadline_day))
                expected_dev = required_dev_m * elapsed_frac
                trajectory_ratio = dev_progress_m / expected_dev if expected_dev > 1e-6 else 1.0
            else:
                trajectory_ratio = 1.0

            if trajectory_ratio < 1.0:
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
    ) -> Optional[Tuple[MissionType, int, int, bool]]:
        """Determines the exact mission assignment for an available haul truck.

        Returns:
            Tuple of (mission_type, target_face_id, target_level, is_waste), or None if truck should remain idle.
        """
        stock_full = current_total_stock >= (self.target_stockpile_buffer_tonnes - 1e-6)

        # -------------------------------------------------------------------
        # Tier 1: Primary Ore Production (When plant needs ore and stockpiles < 60kt)
        # -------------------------------------------------------------------
        if not is_plant_shutdown and not stock_full and can_mine_ore:
            target_face_id = preferred_face_id
            target_face = next((f for f in faces if f.face_id == target_face_id), None)

            if target_face is not None and target_face.state == FaceState.DEVELOPMENT_TURNAROUND:
                # Preferred face is currently in turnaround: muck turnaround waste
                target_level = face_levels.get(target_face_id, default_area1_level)
                return MissionType.STOPE_TURNAROUND_DEV, target_face_id, target_level, True
            elif target_face is not None and target_face.is_ore_available:
                # Preferred face has blasted ore ready: haul ore
                target_level = face_levels.get(
                    target_face_id,
                    default_area2_level if target_face_id == 2 else default_area1_level,
                )
                return MissionType.ORE_HAUL, target_face_id, target_level, False

        # -------------------------------------------------------------------
        # Tier 2: Surplus / Extra Capacity Allocation based on FleetOperatingMode
        # -------------------------------------------------------------------
        # Find if any stope is currently in turnaround
        stope_in_turnaround = next(
            (f for f in faces if f.state == FaceState.DEVELOPMENT_TURNAROUND),
            None,
        )

        if self.current_mode == FleetOperatingMode.PRODUCTION:
            # 1. Stope Turnaround Development first
            if stope_in_turnaround is not None:
                lvl = face_levels.get(stope_in_turnaround.face_id, default_area1_level)
                return MissionType.STOPE_TURNAROUND_DEV, stope_in_turnaround.face_id, lvl, True

            # 2. Capital Decline Development second (if Area 2 is locked)
            if area2_locked:
                return MissionType.CAPITAL_DECLINE_DEV, -1, default_area2_level, True

        elif self.current_mode == FleetOperatingMode.DEVELOPMENT:
            # 1. Capital Decline Development first (if Area 2 is locked)
            if area2_locked:
                return MissionType.CAPITAL_DECLINE_DEV, -1, default_area2_level, True

            # 2. Stope Turnaround Development second
            if stope_in_turnaround is not None:
                lvl = face_levels.get(stope_in_turnaround.face_id, default_area1_level)
                return MissionType.STOPE_TURNAROUND_DEV, stope_in_turnaround.face_id, lvl, True

        # Fallback: If stockpiles are still not full and ore can be mined, continue hauling ore
        if not is_plant_shutdown and not stock_full and can_mine_ore:
            target_face_id = preferred_face_id
            target_level = face_levels.get(
                target_face_id,
                default_area2_level if target_face_id == 2 else default_area1_level,
            )
            return MissionType.ORE_HAUL, target_face_id, target_level, False

        # No active mission needed -> truck remains on standby (idle pacing)
        return None
