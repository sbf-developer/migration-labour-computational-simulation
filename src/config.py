"""Model parameters and stylised calibration targets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

TreatmentId = Literal[
    "baseline",
    "shock_1pct_balanced",
    "shock_3pct_balanced",
    "shock_5pct_balanced",
    "shock_5pct_low_skill",
    "shock_5pct_high_skill",
    "shock_10pct_balanced",
    "shock_5pct_complements",
    "shock_5pct_recession",
]


@dataclass(frozen=True)
class MechanismConfig:
    """Nested mechanism switches M0--M6."""

    local_demand: bool = False
    firm_entry_exit: bool = False
    capital_adjustment: bool = False
    task_specialization: bool = False
    migrant_entrepreneurship: bool = False
    innovation: bool = False

    @classmethod
    def from_level(cls, level: int) -> MechanismConfig:
        if not 0 <= level <= 6:
            raise ValueError("Mechanism level must be in [0, 6]")
        return cls(
            local_demand=level >= 1,
            firm_entry_exit=level >= 2,
            capital_adjustment=level >= 3,
            task_specialization=level >= 4,
            migrant_entrepreneurship=level >= 5,
            innovation=level >= 6,
        )

    @property
    def level(self) -> int:
        flags = [
            self.local_demand,
            self.firm_entry_exit,
            self.capital_adjustment,
            self.task_specialization,
            self.migrant_entrepreneurship,
            self.innovation,
        ]
        return sum(flags)


@dataclass
class SimulationConfig:
    n_regions: int = 5
    n_sectors: int = 3  # low, medium, high skill
    n_tasks: int = 3
    n_firms_per_cell: int = 28
    n_workers: int = 2500
    burn_in_years: int = 10
    post_shock_years: int = 12
    shock_year: int = 10
    shock_size: float = 0.05  # fraction of labour force
    shock_skill: str = "balanced"  # low, medium, high, balanced, complements
    economic_state: str = "normal"  # normal, recession, shortage, inflation
    adjustment_speed: str = "slow"  # fixed, slow, rapid
    migrant_remittance: float = 0.05
    migrant_entrepreneurship_multiplier: float = 1.0
    substitution_tightness: float = 1.0  # higher = closer substitutes
    rng_seed: int = 42
    dt: float = 1.0

    # Production
    capital_share: float = 0.30
    task_elasticity: float = 1.5
    native_migrant_rho: tuple[float, float, float] = (2.5, 2.0, 1.8)
    native_task_weight: tuple[float, float, float] = (0.55, 0.50, 0.45)

    # Matching (Cobb-Douglas matching function)
    matching_efficiency: float = 0.65
    matching_eta: float = 0.55

    # Firm dynamics
    entry_cost: float = 2.5
    exit_loss_threshold: float = -0.4
    investment_rate: float = 0.12

    # Innovation
    innovation_base_prob: float = 0.015
    diversity_elasticity: float = 0.35

    # Spatial
    housing_elasticity: float = 0.25
    migration_cost: float = 0.08


def load_calibration_targets() -> dict:
    path = Path(__file__).resolve().parents[1] / "data" / "calibration_targets.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class CalibratedParams:
    """Parameters anchored to stylised moments from the immigration literature."""

    wage_floor: float = 0.55
    productivity_baseline: float = 1.0
    native_productivity_premium: float = 1.08
    migrant_downgrade_penalty: float = 0.82
    reservation_wage_spread: float = 0.15
    demand_elasticity: float = 0.85
    price_adjustment: float = 0.35
    consumption_mpc: float = 0.78
    tax_rate: float = 0.38
    entrepreneur_entry_gamma: tuple[float, float, float, float] = (
        -1.2,
        1.4,
        0.9,
        0.6,
    )

    @classmethod
    def from_targets(cls) -> CalibratedParams:
        targets = load_calibration_targets()["moments"]
        gap = targets["native_migrant_wage_ratio"]
        return cls(
            native_productivity_premium=gap ** 0.4,
            migrant_downgrade_penalty=1.0 / gap ** 0.35,
            reservation_wage_spread=0.12 + 0.03 * (gap - 1.0),
        )


@dataclass(frozen=True)
class TreatmentSpec:
    name: str
    label: str
    shock_size: float
    shock_skill: str
    economic_state: str = "normal"
    substitution_tightness: float = 1.0
    adjustment_speed: str = "slow"
    migrant_remittance: float = 0.05
    migrant_entrepreneurship_multiplier: float = 1.0


TREATMENTS: dict[TreatmentId, TreatmentSpec] = {
    "baseline": TreatmentSpec("baseline", "No immigration shock", 0.0, "balanced"),
    "shock_1pct_balanced": TreatmentSpec(
        "shock_1pct_balanced", "1\\% balanced inflow", 0.01, "balanced"
    ),
    "shock_3pct_balanced": TreatmentSpec(
        "shock_3pct_balanced", "3\\% balanced inflow", 0.03, "balanced"
    ),
    "shock_5pct_balanced": TreatmentSpec(
        "shock_5pct_balanced", "5\\% balanced inflow", 0.05, "balanced"
    ),
    "shock_5pct_low_skill": TreatmentSpec(
        "shock_5pct_low_skill", "5\\% low-skilled inflow", 0.05, "low"
    ),
    "shock_5pct_high_skill": TreatmentSpec(
        "shock_5pct_high_skill", "5\\% high-skilled inflow", 0.05, "high"
    ),
    "shock_10pct_balanced": TreatmentSpec(
        "shock_10pct_balanced", "10\\% balanced inflow", 0.10, "balanced"
    ),
    "shock_5pct_complements": TreatmentSpec(
        "shock_5pct_complements",
        "5\\% complementary-task inflow",
        0.05,
        "complements",
        substitution_tightness=0.45,
    ),
    "shock_5pct_recession": TreatmentSpec(
        "shock_5pct_recession",
        "5\\% inflow during recession",
        0.05,
        "balanced",
        economic_state="recession",
    ),
}


def skill_distribution(skill: str, rng: np.random.Generator) -> np.ndarray:
    """Return skill shares for three sectors (0=low, 1=mid, 2=high)."""
    if skill == "low":
        probs = np.array([0.70, 0.25, 0.05])
    elif skill == "medium":
        probs = np.array([0.15, 0.60, 0.25])
    elif skill == "high":
        probs = np.array([0.05, 0.20, 0.75])
    elif skill == "complements":
        probs = np.array([0.10, 0.15, 0.75])
    else:  # balanced
        probs = np.array([0.33, 0.34, 0.33])
    return rng.choice(3, size=1, p=probs / probs.sum())


def apply_economic_state(state: str, year: int, shock_year: int) -> dict[str, float]:
    """Multipliers for aggregate conditions."""
    if state == "recession" and year >= shock_year:
        return {"demand": 0.88, "entry": 0.75, "separation": 1.15}
    if state == "shortage" and year >= shock_year:
        return {"demand": 1.12, "entry": 1.20, "separation": 0.90}
    if state == "inflation" and year >= shock_year:
        return {"demand": 1.05, "entry": 0.95, "separation": 1.05}
    return {"demand": 1.0, "entry": 1.0, "separation": 1.0}
