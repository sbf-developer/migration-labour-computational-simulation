"""
Agent-based labour-market model with nested mechanism decomposition (M0--M6).

The model is a stylised computational laboratory calibrated to Denmark-inspired
aggregate moments from published statistics. It is not a literal replay of Danish
administrative microdata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents import FirmArrays, WorkerArrays
from config import (
    CalibratedParams,
    MechanismConfig,
    SimulationConfig,
    TreatmentSpec,
    apply_economic_state,
    load_calibration_targets,
    skill_distribution,
)


def _ces_aggregate(x: np.ndarray, sigma: float) -> float:
    x = np.maximum(x, 1e-8)
    if abs(sigma - 1.0) < 1e-6:
        return float(np.exp(np.mean(np.log(x))))
    exp = (sigma - 1.0) / sigma
    return float(np.sum(x**exp) ** (1.0 / exp))


def _native_migrant_composite(
    n_native: float,
    n_migrant: float,
    theta: float,
    rho: float,
) -> float:
    if n_native + n_migrant < 1e-8:
        return 0.0
    if abs(rho - 1.0) < 1e-6:
        return theta * n_native + (1.0 - theta) * n_migrant
    exp = (rho - 1.0) / rho
    inner = theta * max(n_native, 0.0) ** exp + (1.0 - theta) * max(n_migrant, 0.0) ** exp
    return inner ** (rho / (rho - 1.0))


@dataclass
class LabourMarketModel:
    config: SimulationConfig
    mechanisms: MechanismConfig
    params: CalibratedParams = field(default_factory=CalibratedParams.from_targets)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )
    workers: WorkerArrays = field(init=False)
    firms: FirmArrays = field(init=False)
    sector_demand: np.ndarray = field(init=False)
    regional_rent: np.ndarray = field(init=False)
    next_firm_id: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    pre_shock_native_wage_by_skill: np.ndarray | None = field(default=None, init=False)
    baseline_disposable: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        self.pre_shock_native_wage_by_skill = None
        self._initialise_populations()
        self.sector_demand = np.ones(self.config.n_sectors)
        self.regional_rent = np.full(self.config.n_regions, 1.0)

    def _initialise_populations(self) -> None:
        cfg = self.config
        targets = load_calibration_targets()["moments"]
        n_migrants = int(cfg.n_workers * targets["migrant_share_labor_force"])
        n_natives = cfg.n_workers - n_migrants

        w = WorkerArrays.empty(cfg.n_workers)
        w.native[:n_natives] = True
        w.native[n_natives:] = False

        for i in range(n_natives):
            w.skill[i] = self.rng.integers(0, cfg.n_sectors)
            w.occupation[i] = w.skill[i]
            w.region[i] = self.rng.integers(0, cfg.n_regions)
            w.productivity[i] = self.rng.lognormal(0.0, 0.15) * self.params.native_productivity_premium
            w.reservation_wage[i] = self.params.wage_floor * (1.0 + w.skill[i] * 0.18)
            w.entrepreneurship_propensity[i] = targets["self_employment_rate_native"] * 0.4

        for i in range(n_natives, cfg.n_workers):
            w.skill[i] = self.rng.integers(0, cfg.n_sectors)
            downgrade = self.rng.random() < targets["occupational_downgrade_prob_migrant"]
            w.occupation[i] = max(0, w.skill[i] - 1) if downgrade else w.skill[i]
            w.region[i] = self.rng.integers(0, cfg.n_regions)
            transfer = self.rng.uniform(0.65, 0.95)
            w.skill_transferability[i] = transfer
            w.productivity[i] = (
                self.rng.lognormal(0.0, 0.18)
                * transfer
                * self.params.migrant_downgrade_penalty
            )
            w.reservation_wage[i] = self.params.wage_floor * (0.92 + w.occupation[i] * 0.15)
            w.remittance_rate[i] = cfg.migrant_remittance
            w.entrepreneurship_propensity[i] = (
                targets["self_employment_rate_migrant"] * 0.45 * cfg.migrant_entrepreneurship_multiplier
            )

        self.workers = w

        n_firms = cfg.n_regions * cfg.n_sectors * cfg.n_firms_per_cell
        f = FirmArrays.empty(n_firms, cfg.n_tasks)
        idx = 0
        for r in range(cfg.n_regions):
            for s in range(cfg.n_sectors):
                for _ in range(cfg.n_firms_per_cell):
                    f.sector[idx] = s
                    f.region[idx] = r
                    f.productivity[idx] = self.rng.lognormal(0.0, 0.12)
                    f.capital[idx] = self.rng.uniform(0.8, 1.4)
                    f.price[idx] = 1.0
                    f.expected_demand[idx] = 1.0
                    for q in range(cfg.n_tasks):
                        f.wage_offer[idx, q] = self.params.wage_floor * (1.0 + s * 0.2 + q * 0.05)
                        f.vacancies[idx, q] = 2.5
                    idx += 1
        self.firms = f
        self.next_firm_id = n_firms
        self._assign_initial_jobs()

    def _assign_initial_jobs(self) -> None:
        """Assign workers to firms in the same region; allow adjacent skill sectors."""
        w, f = self.workers, self.firms
        order = self.rng.permutation(w.n)
        for i in order:
            region = int(w.region[i])
            occ = int(w.occupation[i])
            sector_candidates = [occ, max(0, occ - 1), min(self.config.n_sectors - 1, occ + 1)]
            assigned = False
            for s in sector_candidates:
                candidates = np.where(
                    (f.active) & (f.region == region) & (f.sector == s)
                )[0]
                if len(candidates) == 0:
                    continue
                j = int(candidates[self.rng.integers(0, len(candidates))])
                q = int(min(w.occupation[i], self.config.n_tasks - 1))
                if w.native[i]:
                    f.n_native[j, q] += 1
                else:
                    f.n_migrant[j, q] += 1
                w.employed[i] = True
                w.employer_idx[i] = j
                w.wage[i] = f.wage_offer[j, q] * w.productivity[i]
                assigned = True
                break
            if not assigned:
                w.reservation_wage[i] *= 0.95

    def _firm_output(self, j: int) -> float:
        cfg = self.config
        f = self.firms
        if not f.active[j]:
            return 0.0
        task_inputs = []
        s = int(f.sector[j])
        theta = cfg.native_task_weight[s]
        rho = cfg.native_migrant_rho[s] / max(cfg.substitution_tightness, 0.2)
        for q in range(cfg.n_tasks):
            labour_q = _native_migrant_composite(
                f.n_native[j, q], f.n_migrant[j, q], theta, rho
            )
            task_inputs.append(labour_q)
        task_bundle = _ces_aggregate(np.array(task_inputs), cfg.task_elasticity)
        alpha = cfg.capital_share
        return float(
            f.productivity[j] * (max(f.capital[j], 0.05) ** alpha) * (max(task_bundle, 1e-8) ** (1 - alpha))
        )

    def _firm_capacity(self, j: int) -> float:
        return 6.0 + 4.0 * self.firms.capital[j]

    def _firm_staff(self, j: int) -> float:
        f = self.firms
        return float(np.sum(f.n_native[j, :] + f.n_migrant[j, :]))

    def _update_production_and_prices(self, state_mult: dict[str, float]) -> None:
        cfg = self.config
        f = self.firms
        total_output_by_sector = np.zeros(cfg.n_sectors)
        for j in range(f.n):
            if not f.active[j]:
                f.output[j] = 0.0
                continue
            y = self._firm_output(j)
            f.output[j] = y
            total_output_by_sector[int(f.sector[j])] += y

        if self.mechanisms.local_demand:
            for s in range(cfg.n_sectors):
                gap = self.sector_demand[s] - total_output_by_sector[s]
                f.price[:] = np.where(
                    f.active,
                    f.price * (1.0 + self.params.price_adjustment * 0.01 * np.sign(gap)),
                    f.price,
                )
        else:
            f.price[:] = np.where(f.active, 1.0, f.price)

        revenue = f.output * f.price * state_mult["demand"]
        wage_bill = np.sum(f.wage_offer * (f.n_native + f.n_migrant), axis=1)
        f.profits = np.where(f.active, revenue - wage_bill - 0.05 * f.capital, 0.0)

    def _update_household_demand(self) -> None:
        if not self.mechanisms.local_demand:
            self.sector_demand[:] = 1.0
            return
        w = self.workers
        cfg = self.config
        p = self.params
        income = np.where(w.employed, w.wage, 0.0)
        remit = np.where(~w.native, income * w.remittance_rate, 0.0)
        disposable = income * (1.0 - p.tax_rate) - remit - self.regional_rent[w.region] * p.consumption_mpc * 0.1
        disposable = np.maximum(disposable, 0.0)
        w.income = income
        total_disp = float(np.sum(disposable))
        if self.baseline_disposable <= 0:
            self.baseline_disposable = max(total_disp, 1.0)
        demand_scale = np.clip(total_disp / self.baseline_disposable, 0.85, 1.25)
        shares = np.array(
            [
                load_calibration_targets()["moments"]["consumption_share_low_skill_sector"],
                load_calibration_targets()["moments"]["consumption_share_mid_skill_sector"],
                load_calibration_targets()["moments"]["consumption_share_high_skill_sector"],
            ]
        )
        self.sector_demand = demand_scale * shares / max(shares.sum(), 1e-8)
        self.sector_demand *= self.config.n_sectors  # index around 1

    def _labour_market_tightness(self, region: int, sector: int) -> float:
        w = self.workers
        in_cell = (w.region == region) & ((w.occupation == sector) | (w.skill == sector))
        unemployed = in_cell & (~w.employed)
        employed = in_cell & w.employed
        u = float(np.sum(unemployed))
        e = float(np.sum(employed))
        return u / max(u + e, 1.0)

    def _post_vacancies(self, state_mult: dict[str, float]) -> None:
        cfg = self.config
        f = self.firms
        fixed = cfg.adjustment_speed == "fixed" or not self.mechanisms.firm_entry_exit
        adj = 0.0 if fixed else (1.0 if cfg.adjustment_speed == "rapid" else 0.45)
        for j in range(f.n):
            if not f.active[j]:
                continue
            s = int(f.sector[j])
            r = int(f.region[j])
            demand_signal = self.sector_demand[s] if self.mechanisms.local_demand else 1.0
            util = f.output[j] / max(f.expected_demand[j], 0.1)
            tightness = self._labour_market_tightness(r, s)
            for q in range(cfg.n_tasks):
                target = 1.0 + adj * (0.5 * (demand_signal - 1.0) + 0.3 * (util - 1.0))
                if fixed:
                    target = 1.0
                f.vacancies[j, q] = np.clip(target, 1.0 if fixed else 0.0, 4.0)
                wage_pressure = 0.12 * (tightness - 0.08) - 0.08 * (util - 1.0) * adj
                f.wage_offer[j, q] = np.clip(
                    f.wage_offer[j, q] * (1.0 - wage_pressure),
                    self.params.wage_floor * (0.85 + 0.08 * s),
                    self.params.wage_floor * (1.35 + 0.15 * s),
                )
            f.expected_demand[j] = 0.85 * f.expected_demand[j] + 0.15 * f.output[j] * state_mult["demand"]

    def _matching_and_separations(self, state_mult: dict[str, float]) -> None:
        cfg = self.config
        w, f = self.workers, self.firms
        sep_rate = load_calibration_targets()["moments"]["separation_rate_quarterly"] * 2.0
        sep_rate *= state_mult["separation"]

        employed_idx = np.where(w.employed)[0]
        sep_draw = self.rng.random(len(employed_idx)) < sep_rate
        for i, sep in zip(employed_idx, sep_draw):
            if not sep:
                continue
            j = int(w.employer_idx[i])
            q = int(min(w.occupation[i], cfg.n_tasks - 1))
            if w.native[i]:
                f.n_native[j, q] = max(0.0, f.n_native[j, q] - 1.0)
            else:
                f.n_migrant[j, q] = max(0.0, f.n_migrant[j, q] - 1.0)
            w.employed[i] = False
            w.employer_idx[i] = -1
            w.wage[i] = 0.0

        unemployed = np.where(~w.employed)[0]
        self.rng.shuffle(unemployed)
        for i in unemployed:
            region = int(w.region[i])
            occ = int(w.occupation[i])
            sector_options = [occ]
            if self.rng.random() < 0.25:
                sector_options.append(max(0, occ - 1))
            if self.rng.random() < 0.25:
                sector_options.append(min(cfg.n_sectors - 1, occ + 1))
            matched = False
            for s in sector_options:
                candidates = np.where(
                    (f.active) & (f.region == region) & (f.sector == s) & (np.sum(f.vacancies, axis=1) > 0.05)
                )[0]
                if len(candidates) == 0:
                    continue
                wages = f.wage_offer[candidates, min(occ, cfg.n_tasks - 1)]
                probs = wages / max(wages.sum(), 1e-8)
                j = int(candidates[self.rng.choice(len(candidates), p=probs)])
                q = min(occ, cfg.n_tasks - 1)
                if f.vacancies[j, q] < 0.05:
                    continue
                if self._firm_staff(j) >= self._firm_capacity(j):
                    continue
                f.vacancies[j, q] -= 1.0
                if w.native[i]:
                    f.n_native[j, q] += 1.0
                else:
                    f.n_migrant[j, q] += 1.0
                w.employed[i] = True
                w.employer_idx[i] = j
                offer = f.wage_offer[j, q]
                if not w.native[i]:
                    offer *= 0.96
                w.wage[i] = offer * w.productivity[i]
            # Employed natives in same firm/sector feel competition via wage compression
            if w.native[i] and not self.mechanisms.local_demand:
                peer_mask = w.employed & (w.employer_idx == j) & w.native
                w.wage[peer_mask] *= 0.998
                matched = True
                break

    def _firm_entry_exit(self, state_mult: dict[str, float]) -> None:
        if not self.mechanisms.firm_entry_exit:
            return
        if self.config.adjustment_speed == "fixed":
            return
        cfg = self.config
        f = self.firms
        g0, g1, g2, g3 = self.params.entrepreneur_entry_gamma
        speed = 1.0 if cfg.adjustment_speed == "rapid" else 0.45

        for j in range(f.n):
            if not f.active[j]:
                continue
            if f.profits[j] < self.config.exit_loss_threshold:
                if self.rng.random() < 0.04 * speed:
                    f.active[j] = False
                    f.n_native[j, :] = 0.0
                    f.n_migrant[j, :] = 0.0
                    self._layoff_firm_workers(j)

        for r in range(cfg.n_regions):
            for s in range(cfg.n_sectors):
                active_mask = (f.active) & (f.region == r) & (f.sector == s)
                if not np.any(active_mask):
                    avg_profit = 0.2
                else:
                    avg_profit = float(np.mean(f.profits[active_mask]))
                demand = self.sector_demand[s] if self.mechanisms.local_demand else 1.0
                index = g0 + g1 * avg_profit - g2 * self.config.entry_cost + g3 * demand
                p_entry = 1.0 / (1.0 + np.exp(-index))
                p_entry *= speed * state_mult["entry"]
                if self.rng.random() < p_entry * 0.08:
                    self._create_firm(r, s, migrant_owned=False)

        if self.mechanisms.migrant_entrepreneurship:
            migrants = np.where(~self.workers.native)[0]
            for i in migrants:
                if self.workers.employed[i]:
                    continue
                prop = self.workers.entrepreneurship_propensity[i]
                if self.rng.random() < prop * speed * 0.15:
                    r = int(self.workers.region[i])
                    s = int(self.workers.occupation[i])
                    self._create_firm(r, s, migrant_owned=True)

    def _layoff_firm_workers(self, j: int) -> None:
        w = self.workers
        mask = w.employed & (w.employer_idx == j)
        w.employed[mask] = False
        w.employer_idx[mask] = -1
        w.wage[mask] = 0.0

    def _create_firm(self, region: int, sector: int, migrant_owned: bool) -> None:
        f = self.firms
        inactive = np.where(~f.active)[0]
        if len(inactive) == 0:
            return
        j = int(inactive[0])
        f.active[j] = True
        f.region[j] = region
        f.sector[j] = sector
        f.productivity[j] = self.rng.lognormal(0.0, 0.15) * (0.92 if migrant_owned else 1.0)
        f.capital[j] = self.rng.uniform(0.5, 1.0)
        f.n_native[j, :] = 0.0
        f.n_migrant[j, :] = 0.0
        f.vacancies[j, :] = 2.0
        f.wage_offer[j, :] = self.params.wage_floor * (1.0 + sector * 0.18)
        f.age[j] = 0
        f.profits[j] = 0.0
        f.output[j] = 0.0

    def _capital_adjustment(self) -> None:
        if not self.mechanisms.capital_adjustment:
            return
        if self.config.adjustment_speed == "fixed":
            return
        f = self.firms
        rate = self.config.investment_rate
        if self.config.adjustment_speed == "rapid":
            rate *= 1.6
        for j in range(f.n):
            if not f.active[j]:
                continue
            util = f.output[j] / max(f.expected_demand[j], 0.1)
            invest = rate * max(f.profits[j], 0.0) * (0.8 + 0.4 * util)
            f.capital[j] = max(0.2, f.capital[j] + invest - 0.03 * f.capital[j])

    def _task_specialisation(self) -> None:
        if not self.mechanisms.task_specialization:
            return
        w = self.workers
        for i in range(w.n):
            if not w.employed[i] or not w.native[i]:
                continue
            if w.occupation[i] >= w.skill[i]:
                continue
            if self.rng.random() < 0.06:
                w.occupation[i] = min(w.skill[i], self.config.n_sectors - 1)
                w.wage[i] *= 1.04

    def _innovation(self) -> None:
        if not self.mechanisms.innovation:
            return
        cfg = self.config
        f = self.firms
        w = self.workers
        for j in range(f.n):
            if not f.active[j]:
                continue
            region = int(f.region[j])
            in_region = w.region == region
            diversity = float(np.mean(w.skill[in_region])) / max(cfg.n_sectors - 1, 1)
            high_skill_share = float(np.mean(w.skill[in_region] >= 2))
            rd_labor = float(np.sum(f.n_native[j, :]) + np.sum(f.n_migrant[j, :]))
            spillover = float(np.mean(f.productivity[f.active & (f.region == region)]))
            prob = (
                cfg.innovation_base_prob
                * (1.0 + cfg.diversity_elasticity * diversity)
                * (1.0 + 0.2 * high_skill_share)
                * (1.0 + 0.05 * np.log1p(rd_labor))
                * (1.0 + 0.08 * (spillover - 1.0))
            )
            if self.rng.random() < prob:
                f.productivity[j] *= self.rng.uniform(1.02, 1.08)
                # Innovation raises marginal product of labour → wage offers follow
                f.wage_offer[j, :] *= 1.01

    def _spatial_adjustment(self) -> None:
        cfg = self.config
        w = self.workers
        for r in range(cfg.n_regions):
            emp_rate = float(np.mean(w.employed[w.region == r]))
            self.regional_rent[r] = 1.0 + cfg.housing_elasticity * (emp_rate - 0.75)

        mobile = ~w.native | (self.rng.random(w.n) < 0.15)
        for i in range(w.n):
            if not mobile[i] or self.rng.random() > 0.08:
                continue
            best_r = int(w.region[i])
            best_u = float(np.mean(w.employed[w.region == best_r]))
            for r in range(cfg.n_regions):
                emp = w.employed[w.region == r]
                if len(emp) == 0:
                    continue
                avg_wage = float(np.mean(w.wage[w.region == r][emp])) if np.any(emp) else 0.0
                score = avg_wage * float(np.mean(emp)) - cfg.migration_cost - 0.05 * self.regional_rent[r]
                base = float(np.mean(w.wage[w.region == best_r][w.employed[w.region == best_r]])) if best_u > 0 else 0.0
                base_score = base * best_u - 0.05 * self.regional_rent[best_r]
                if score > base_score:
                    best_r = r
            w.region[i] = best_r

    def inject_immigration_shock(self, treatment: TreatmentSpec) -> None:
        cfg = self.config
        n_shock = int(cfg.n_workers * treatment.shock_size)
        if n_shock <= 0:
            return
        old_n = self.workers.n
        new_n = old_n + n_shock
        w_new = WorkerArrays.empty(new_n)
        for field_name in (
            "native", "skill", "occupation", "productivity", "reservation_wage",
            "employed", "employer_idx", "wage", "region", "remittance_rate",
            "skill_transferability", "entrepreneurship_propensity", "income",
        ):
            old = getattr(self.workers, field_name)
            new = getattr(w_new, field_name)
            new[:old_n] = old

        targets = load_calibration_targets()["moments"]
        for i in range(old_n, new_n):
            w_new.native[i] = False
            sk = int(skill_distribution(treatment.shock_skill, self.rng)[0])
            w_new.skill[i] = sk
            if treatment.shock_skill == "complements":
                w_new.occupation[i] = min(sk + 1, cfg.n_sectors - 1) if sk < 2 else 2
            else:
                downgrade = self.rng.random() < targets["occupational_downgrade_prob_migrant"]
                w_new.occupation[i] = max(0, sk - 1) if downgrade else sk
            w_new.region[i] = self.rng.integers(0, cfg.n_regions)
            transfer = self.rng.uniform(0.60, 0.95)
            w_new.skill_transferability[i] = transfer
            w_new.productivity[i] = (
                self.rng.lognormal(0.0, 0.18) * transfer * self.params.migrant_downgrade_penalty
            )
            w_new.reservation_wage[i] = self.params.wage_floor * (0.90 + w_new.occupation[i] * 0.14)
            w_new.remittance_rate[i] = treatment.migrant_remittance
            w_new.entrepreneurship_propensity[i] = (
                targets["self_employment_rate_migrant"] * 0.45 * treatment.migrant_entrepreneurship_multiplier
            )
        self.workers = w_new
        self.config = SimulationConfig(**{**cfg.__dict__, "shock_size": treatment.shock_size})

        # Immigration boosts local demand when consumption channel is active
        if self.mechanisms.local_demand:
            self.baseline_disposable *= 1.0 + 0.4 * treatment.shock_size * (1.0 - treatment.migrant_remittance)
        f = self.firms
        # Immediate competition effect on wage offers in affected sectors
        comp = 0.85 * treatment.shock_size * cfg.substitution_tightness
        for j in range(f.n):
            if not f.active[j]:
                continue
            s = int(f.sector[j])
            migrant_share = float(np.mean(~self.workers.native[self.workers.skill == s]))
            f.wage_offer[j, :] *= max(0.88, 1.0 - comp * migrant_share)

        if self.mechanisms.innovation:
            for j in range(f.n):
                if f.active[j]:
                    f.productivity[j] *= 1.0 + 0.02 * treatment.shock_size

    def _collect_metrics(self, year: int) -> dict[str, Any]:
        w, f = self.workers, self.firms
        native = w.native
        migrant = ~native
        metrics: dict[str, Any] = {"year": year}

        def avg_wage(mask: np.ndarray) -> float:
            emp = mask & w.employed
            return float(np.mean(w.wage[emp])) if np.any(emp) else np.nan

        metrics["native_wage"] = avg_wage(native)
        metrics["migrant_wage"] = avg_wage(migrant)
        for sk in range(self.config.n_sectors):
            mask = native & (w.skill == sk)
            metrics[f"native_wage_skill_{sk}"] = avg_wage(mask)
            metrics[f"native_employment_skill_{sk}"] = float(np.mean(w.employed[mask])) if np.any(mask) else np.nan

        metrics["native_employment"] = float(np.mean(w.employed[native]))
        metrics["migrant_employment"] = float(np.mean(w.employed[migrant]))
        metrics["unemployment_rate"] = float(np.mean(~w.employed))
        metrics["vacancy_rate"] = float(np.mean(f.vacancies[f.active]))
        metrics["firm_count"] = int(np.sum(f.active))
        metrics["output_per_capita"] = float(np.sum(f.output[f.active]) / max(w.n, 1))
        metrics["avg_productivity"] = float(np.mean(f.productivity[f.active]))
        metrics["firm_entry_stock"] = int(np.sum(f.active))
        metrics["price_index"] = float(np.mean(f.price[f.active]))
        metrics["profit_rate"] = float(np.mean(f.profits[f.active]))

        if self.pre_shock_native_wage_by_skill is not None:
            changes = []
            for sk in range(self.config.n_sectors):
                key = f"native_wage_skill_{sk}"
                base = self.pre_shock_native_wage_by_skill[sk]
                if base > 0 and not np.isnan(metrics[key]):
                    changes.append((metrics[key] - base) / base * 100.0)
            metrics["native_wage_index_change_pct"] = float(np.mean(changes)) if changes else np.nan
        return metrics

    def run(self, treatment: TreatmentSpec | None = None) -> list[dict[str, Any]]:
        cfg = self.config
        total_years = cfg.burn_in_years + cfg.post_shock_years
        self.history = []
        pre_shock_native_wage = None

        for year in range(total_years):
            if year == cfg.shock_year and treatment is not None and treatment.shock_size > 0:
                pre = self._collect_metrics(year)
                pre_shock_native_wage = pre["native_wage"]
                self.pre_shock_native_wage_by_skill = np.array(
                    [pre[f"native_wage_skill_{sk}"] for sk in range(cfg.n_sectors)]
                )
                w = self.workers
                self.baseline_disposable = max(float(np.sum(w.income)), 1.0) if np.any(w.income) else 1.0
                self.inject_immigration_shock(treatment)

            state_mult = apply_economic_state(
                treatment.economic_state if treatment else "normal",
                year,
                cfg.shock_year,
            )

            self._update_household_demand()
            self._post_vacancies(state_mult)
            self._matching_and_separations(state_mult)
            if year == cfg.shock_year:
                self._matching_and_separations(state_mult)
            self._update_production_and_prices(state_mult)
            self._firm_entry_exit(state_mult)
            self._capital_adjustment()
            self._task_specialisation()
            self._innovation()
            if year % 4 == 0:
                self._spatial_adjustment()

            metrics = self._collect_metrics(year)
            if pre_shock_native_wage is not None and year >= cfg.shock_year:
                metrics["native_wage_change_pct"] = (
                    (metrics["native_wage"] - pre_shock_native_wage) / pre_shock_native_wage * 100.0
                )
                if "native_wage_index_change_pct" in metrics:
                    metrics["native_wage_change_short_index"] = metrics["native_wage_index_change_pct"]
            self.history.append(metrics)

        return self.history
