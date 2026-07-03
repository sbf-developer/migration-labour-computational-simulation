"""Agent state containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorkerArrays:
    """Vectorised worker population for computational efficiency."""

    n: int
    native: np.ndarray  # bool
    skill: np.ndarray  # int 0..2
    occupation: np.ndarray  # int 0..2
    productivity: np.ndarray
    reservation_wage: np.ndarray
    employed: np.ndarray  # bool
    employer_idx: np.ndarray  # int, -1 if unemployed
    wage: np.ndarray
    region: np.ndarray  # int
    remittance_rate: np.ndarray
    skill_transferability: np.ndarray
    entrepreneurship_propensity: np.ndarray
    income: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def empty(cls, n: int) -> WorkerArrays:
        return cls(
            n=n,
            native=np.ones(n, dtype=bool),
            skill=np.zeros(n, dtype=np.int8),
            occupation=np.zeros(n, dtype=np.int8),
            productivity=np.ones(n),
            reservation_wage=np.full(n, 0.6),
            employed=np.zeros(n, dtype=bool),
            employer_idx=np.full(n, -1, dtype=np.int32),
            wage=np.zeros(n),
            region=np.zeros(n, dtype=np.int8),
            remittance_rate=np.zeros(n),
            skill_transferability=np.ones(n),
            entrepreneurship_propensity=np.full(n, 0.02),
            income=np.zeros(n),
        )


@dataclass
class FirmArrays:
    """Firm population with task-level staffing."""

    n: int
    n_tasks: int
    active: np.ndarray
    sector: np.ndarray
    region: np.ndarray
    productivity: np.ndarray
    capital: np.ndarray
    n_native: np.ndarray  # (n_firms, n_tasks)
    n_migrant: np.ndarray
    vacancies: np.ndarray
    wage_offer: np.ndarray
    price: np.ndarray
    expected_demand: np.ndarray
    profits: np.ndarray
    output: np.ndarray
    age: np.ndarray

    @classmethod
    def empty(cls, n: int, n_tasks: int) -> FirmArrays:
        return cls(
            n=n,
            n_tasks=n_tasks,
            active=np.ones(n, dtype=bool),
            sector=np.zeros(n, dtype=np.int8),
            region=np.zeros(n, dtype=np.int8),
            productivity=np.ones(n),
            capital=np.ones(n),
            n_native=np.zeros((n, n_tasks)),
            n_migrant=np.zeros((n, n_tasks)),
            vacancies=np.zeros((n, n_tasks)),
            wage_offer=np.full((n, n_tasks), 0.7),
            price=np.ones(n),
            expected_demand=np.ones(n),
            profits=np.zeros(n),
            output=np.zeros(n),
            age=np.zeros(n, dtype=np.int16),
        )
