#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Asigna coordenadas a una tabla de avistamientos de aves usando un CSV maestro de coordenadas.

Reglas:
- Si la especie aparece una sola vez en la tabla, se le asigna UNA coordenada aleatoria desde el maestro.
- Si aparece varias veces, cada fila recibe una coordenada aleatoria desde el maestro (con reemplazo).
- Coincidencia por nombre ignorando lo que está entre paréntesis y sin distinguir mayúsculas/minúsculas.
- Columnas añadidas: lat, lopnong, lat_lon ("lat,lon").
- Filas sin match quedan con columnas vacías. Se exporta un reporte AJ/no_match.csv.

Uso (con defaults tomando la estructura de tu carpeta AJ):
python asignar_coordenadas_aves.py \
  --tabla "AJ/Tabla.csv" \
  --maestro "AJ/Cordenadas.CSV" \
  --out "AJ/Tabla_con_coordenadas.csv" \
  --seed 20251003
"""

from __future__ import annotations
import argparse
import pathlib
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def normalize_species(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name)
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)  # quitar paréntesis y contenido
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    norm_map = {c: re.sub(r"\s+", " ", c).strip().casefold() for c in df.columns}
    targets = [t.casefold() for t in candidates]
    for col, norm in norm_map.items():
        if norm in targets:
            return col
    raise KeyError(f"No se encontró ninguna de las columnas esperadas: {candidates} en {list(df.columns)}")


def load_master(master_path: pathlib.Path) -> Dict[str, List[Tuple[float, float]]]:
    dfm = pd.read_csv(master_path, encoding="utf-8")
    col_nombre = find_col(dfm, ["nombre de ave", "nombre_de_ave", "nombre", "especie", "species"])
    col_lat = find_col(dfm, ["latitud", "lat"])
    col_lon = find_col(dfm, ["longitud", "lon", "long", "lopnong"])  # tolera error tipográfico

    dfm["_nombre_norm"] = dfm[col_nombre].apply(normalize_species)

    dfm = dfm.copy()
    dfm["_lat"] = pd.to_numeric(dfm[col_lat], errors="coerce")
    dfm["_lon"] = pd.to_numeric(dfm[col_lon], errors="coerce")
    dfm = dfm.dropna(subset=["_nombre_norm", "_lat", "_lon"])

    master: Dict[str, List[Tuple[float, float]]] = {}
    for name, sub in dfm.groupby("_nombre_norm", dropna=True):
        coords = list(zip(sub["_lat"].astype(float).tolist(), sub["_lon"].astype(float).tolist()))
        if coords:
            master[name] = coords
    return master


def assign_coords(
    df: pd.DataFrame,
    species_col: str,
    master: Dict[str, List[Tuple[float, float]]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    out["_species_norm"] = out[species_col].apply(normalize_species)

    out["lat"] = np.nan
    out["lopnong"] = np.nan

    for sp, idx in out.groupby("_species_norm").groups.items():
        idx_list = list(idx)
        coords_list = master.get(sp, None)
        if not coords_list:
            continue

        if len(idx_list) == 1:
            lat, lon = coords_list[rng.integers(0, len(coords_list))]
            out.loc[idx_list[0], "lat"] = lat
            out.loc[idx_list[0], "lopnong"] = lon
        else:
            choices = rng.integers(0, len(coords_list), size=len(idx_list))
            for row_i, choice in zip(idx_list, choices):
                lat, lon = coords_list[choice]
                out.loc[row_i, "lat"] = lat
                out.loc[row_i, "lopnong"] = lon

    def fmt_pair(a, b) -> str:
        if pd.isna(a) or pd.isna(b):
            return ""
        try:
            return f"{float(a):.6f},{float(b):.6f}"
        except Exception:
            return ""

    out["lat_lon"] = [fmt_pair(a, b) for a, b in zip(out["lat"], out["lopnong"])]
    out["lat"] = out["lat"].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    out["lopnong"] = out["lopnong"].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")

    out = out.drop(columns=["_species_norm"], errors="ignore")
    return out


def main():
    parser = argparse.ArgumentParser(description="Asignar coordenadas a una tabla de aves desde un CSV maestro de coordenadas.")
    parser.add_argument("--tabla", type=str, default=str(pathlib.Path("AJ") / "Tabla.csv"),
                        help="Ruta del CSV de la tabla general de avistamientos. Default: AJ/Tabla.csv")
    parser.add_argument("--maestro", type=str, default=str(pathlib.Path("AJ") / "Cordenadas.csv"),
                        help="Ruta del CSV maestro de coordenadas. Default: AJ/Cordenadas.CSV")
    parser.add_argument("--out", type=str, default=str(pathlib.Path("AJ") / "Tabla_con_coordenadas.csv"),
                        help="Ruta del CSV de salida. Default: AJ/Tabla_con_coordenadas.csv")
    parser.add_argument("--seed", type=int, default=20251003, help="Semilla aleatoria para reproducibilidad.")
    args = parser.parse_args()

    tabla_path = pathlib.Path(args.tabla)
    maestro_path = pathlib.Path(args.maestro)
    out_path = pathlib.Path(args.out)
    no_match_path = out_path.parent / "no_match.csv"

    df = pd.read_csv(tabla_path, encoding="utf-8")
    species_col = find_col(df, ["species", "especie", "nombre de ave", "nombre"])

    master = load_master(maestro_path)

    rng = np.random.default_rng(args.seed)
    result = assign_coords(df, species_col, master, rng)

    df["_species_norm"] = df[species_col].apply(normalize_species)
    species_in_table = set(df["_species_norm"].unique().tolist())
    no_match = sorted([s for s in species_in_table if s and s not in master])
    if no_match:
        pd.DataFrame({"species_norm": no_match}).to_csv(no_match_path, index=False, encoding="utf-8")

    result.to_csv(out_path, index=False, encoding="utf-8")

    total = len(df)
    asignadas = (result["lat"] != "").sum()
    sin_match = len(no_match)
    print(f"Filas totales: {total}")
    print(f"Filas con coordenadas asignadas: {asignadas}")
    print(f"Especies sin match en maestro: {sin_match} -> {no_match_path if sin_match else 'N/A'}")
    print(f"Salida: {out_path}")


if __name__ == "__main__":
    main()
