# toolbox_recommender.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import re

import pandas as pd
import yaml


# -----------------------------
# Load solutions
# -----------------------------
def load_toolbox(yaml_path: str) -> List[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sols = data.get("solutions", [])
    if not isinstance(sols, list):
        raise ValueError("YAML verwacht key 'solutions' als lijst.")
    return sols


# -----------------------------
# Simple + safe condition evaluator
# - ondersteunt: > < >= <= == != and or true false
# - variabelen komen uit cluster_context dict
# -----------------------------
_ALLOWED = re.compile(r"^[A-Za-z0-9_ .<>=!()\-+*/&|]+$")

def _to_bool_literals(expr: str) -> str:
    # YAML gebruikt vaak true/false
    return (
        expr.replace(" true", " True")
            .replace(" false", " False")
            .replace("== true", "== True")
            .replace("== false", "== False")
    )

def eval_condition(expr: str, ctx: Dict[str, Any]) -> bool:
    expr = expr.strip()
    expr = _to_bool_literals(expr)

    # heel simpele input-safety: alleen toegestane chars
    if not _ALLOWED.match(expr):
        raise ValueError(f"Onveilige/ongekende expressie: {expr}")

    # maak eval omgeving dicht
    safe_globals = {"__builtins__": {}}
    safe_locals = dict(ctx)

    try:
        out = eval(expr, safe_globals, safe_locals)
        return bool(out)
    except NameError:
        # variabele ontbreekt -> behandel als 'onbekend' => condition false (dus niet uitsluiten)
        return False
    except Exception:
        # elke andere parse error: condition niet toepassen
        return False


# -----------------------------
# Cluster context builder
# -----------------------------
def bbox_diag_m(row: pd.Series) -> Optional[float]:
    needed = ["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]
    if not all(k in row for k in needed):
        return None
    try:
        dx = float(row["bbox_maxx"]) - float(row["bbox_minx"])
        dy = float(row["bbox_maxy"]) - float(row["bbox_miny"])
        return float(math.sqrt(dx * dx + dy * dy))
    except Exception:
        return None


def build_cluster_context(
    cluster_row: pd.Series,
    # toggles uit de UI (globaal of per cluster)
    grid_constraints_present: bool,
    feed_in_capacity_available: bool,
    flexible_load_present: bool,
    physical_space_available: bool,
    suitable_wind_location_present: bool = False,
    spatial_opportunity_available: bool = True,
    sufficient_thermal_demand_present: bool = False,
    thermal_demand_is_low: bool = False,
) -> Dict[str, Any]:
    n_buildings = int(cluster_row.get("n_Bouwwerk", 0) or 0)

    spread = bbox_diag_m(cluster_row)
    if spread is None:
        spread = float(cluster_row.get("mean_pairwise_dist_m", 0.0) or 0.0)

    vrije_kw = float(cluster_row.get("vrije_ruimte_kw", 0.0) or 0.0)

    required_power_kw = max(0.0, -vrije_kw)
    required_power_per_building_kw = (required_power_kw / n_buildings) if n_buildings > 0 else None

    ctx = {
        # YAML variabelen
        "number_of_buildings": n_buildings,
        "spatial_spread_m": float(spread),
        "required_power_kw": float(required_power_kw),
        "required_power_per_building_kw": float(required_power_per_building_kw) if required_power_per_building_kw is not None else None,
        "electricity_demand_present": (n_buildings > 0),
        "grid_constraints_present": bool(grid_constraints_present),
        "feed_in_capacity_available": bool(feed_in_capacity_available),
        "flexible_load_present": bool(flexible_load_present),
        "physical_space_available": bool(physical_space_available),
        "suitable_wind_location_present": bool(suitable_wind_location_present),
        "spatial_opportunity_available": bool(spatial_opportunity_available),
        "sufficient_thermal_demand_present": bool(sufficient_thermal_demand_present),
        "thermal_demand_is_low": bool(thermal_demand_is_low),

        # extra (handig)
        "vrije_ruimte_kw": float(vrije_kw),
        "mean_pairwise_dist_m": float(cluster_row.get("mean_pairwise_dist_m", 0.0) or 0.0),
        "mean_opwek_to_others_m": float(cluster_row.get("mean_opwek_to_others_m", 0.0) or 0.0),
        "n_Energieproject": int(cluster_row.get("n_Energieproject", 0) or 0),
    }
    return ctx


# -----------------------------
# Applicability + scoring
# -----------------------------
@dataclass
class Recommendation:
    solution_id: str
    name: str
    applicable: bool
    reason: str
    score: float
    solution: dict


def evaluate_solution(ctx: Dict[str, Any], sol: dict) -> Tuple[bool, str]:
    # Preconditions: als precondition in YAML staat, moet ctx[precondition] True zijn
    for pre in sol.get("preconditions", []):
        if not ctx.get(pre, False):
            return False, f"Voorwaarde niet gehaald: {pre}"

    # Exclusion rules: als condition True -> uitsluiten
    for rule in sol.get("exclusion_rules", []):
        cond = rule.get("condition", "")
        if cond and eval_condition(cond, ctx):
            return False, rule.get("reason", "Uitgesloten door regel")

    return True, "Toepasbaar"


def score_solution(ctx: Dict[str, Any], sol: dict) -> float:
    """
    Simpele, uitlegbare score:
    - start basis 0
    - bonus als oplossing matcht met type congestie/logica:
      * tekort (required_power_kw > 0) => afnamecongestie-achtige oplossingen hoger
      * overschot/vrije_ruimte positief + opwek aanwezig => teruglever-achtige oplossingen hoger
    - bonus voor 'structurele' oplossingen bij grote clusters (veel gebouwen)
    """
    s = 0.0

    required_power_kw = float(ctx.get("required_power_kw", 0.0) or 0.0)
    vrije_kw = float(ctx.get("vrije_ruimte_kw", 0.0) or 0.0)
    n_build = int(ctx.get("number_of_buildings", 0) or 0)
    n_energy = int(ctx.get("n_Energieproject", 0) or 0)

    congestion_type = ((sol.get("category") or {}).get("congestion_type") or "").lower()

    # tekort (afname-probleem)
    if required_power_kw > 0:
        if "afname" in congestion_type:
            s += 2.0
        if sol.get("id") in {"battery_storage", "peak_shaving_load_shifting", "contractual_flexibility"}:
            s += 1.0

    # overschot/opwek (teruglever)
    if (vrije_kw > 0) and (n_energy > 0):
        if "teruglever" in congestion_type or "afname_en_teruglevering" in congestion_type:
            s += 2.0
        if sol.get("id") in {"battery_storage", "energy_hub", "urban_battery"}:
            s += 1.0

    # grotere clusters -> hubs/collectief interessanter
    if n_build >= 5 and sol.get("id") in {"energy_hub", "urban_battery", "seasonal_thermal_storage"}:
        s += 2.0

    # iets van “compact = makkelijker samen”
    spread = float(ctx.get("spatial_spread_m", 0.0) or 0.0)
    if spread > 0:
        s += max(0.0, 1.0 - (spread / 2000.0))  # binnen ~2km kleine bonus

    return float(s)


def recommend_for_cluster(ctx: Dict[str, Any], solutions: List[dict], top_k: int = 5) -> List[Recommendation]:
    recs: List[Recommendation] = []
    for sol in solutions:
        applicable, reason = evaluate_solution(ctx, sol)
        sc = score_solution(ctx, sol) if applicable else -999.0
        recs.append(
            Recommendation(
                solution_id=str(sol.get("id")),
                name=str(sol.get("name")),
                applicable=bool(applicable),
                reason=str(reason),
                score=float(sc),
                solution=sol,
            )
        )

    recs = [r for r in recs if r.applicable]
    recs.sort(key=lambda r: r.score, reverse=True)
    return recs[:top_k]
