# validators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd

try:
    from shapely import wkt
except Exception:  # pragma: no cover
    wkt = None


@dataclass
class ValidationResult:
    errors: List[str]
    warnings: List[str]
    info: List[str]


def validate_dataframe(df: pd.DataFrame, *, geometry_col: str = "geometry", source_col: str = "source") -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    # Required columns (based on clustering_pipeline expectations)
    if geometry_col not in df.columns:
        errors.append(f"Verplichte kolom ontbreekt: '{geometry_col}'. Voeg deze kolom toe (WKT punten).")
    if source_col not in df.columns:
        errors.append(f"Verplichte kolom ontbreekt: '{source_col}'. Voeg deze kolom toe (bron/categorie per locatie).")

    if df.empty:
        errors.append("De CSV bevat geen rijen (leeg bestand).")

    # Basic empties
    if geometry_col in df.columns:
        n_null = int(df[geometry_col].isna().sum())
        if n_null > 0:
            errors.append(f"Kolom '{geometry_col}' bevat {n_null} lege waarden. Vul deze aan of verwijder de rijen.")

    # Try parse a small sample of geometry as WKT to catch obvious issues early
    if wkt is None:
        warnings.append("Kon WKT-parser niet laden. Geometrie-validatie is beperkt.")
    elif geometry_col in df.columns and not df.empty:
        sample = df[geometry_col].dropna().astype(str).head(50)
        bad = 0
        for s in sample:
            try:
                wkt.loads(s)
            except Exception:
                bad += 1
        if bad > 0:
            errors.append(
                f"Geometrie lijkt ongeldig: {bad} van de eerste {len(sample)} geometrieën kon niet worden gelezen als WKT. "
                "Controleer dat 'geometry' WKT-punten bevat zoals: POINT (x y)."
            )

    # Non-fatal signals
    if df.shape[0] > 0:
        info.append(f"Ingelezen rijen: {df.shape[0]:,}".replace(",", "."))
        info.append(f"Aantal kolommen: {df.shape[1]}")

    # Helpful warnings for common columns used later (not strictly required)
    optional_cols = ["vrije_ruimte_kw", "n_Bouwwerk", "bbox_minx", "bbox_maxx", "bbox_miny", "bbox_maxy"]
    missing_optional = [c for c in optional_cols if c not in df.columns]
    if missing_optional:
        warnings.append(
            "Sommige kolommen voor detailanalyse ontbreken (niet fataal): "
            + ", ".join(missing_optional)
            + ". De tool kan nog steeds clusteren, maar sommige details/filters kunnen minder informatief zijn."
        )

    return ValidationResult(errors=errors, warnings=warnings, info=info)
