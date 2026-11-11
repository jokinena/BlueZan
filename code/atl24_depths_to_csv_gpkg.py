#!/usr/bin/env python3
"""Extract photon-level bathymetric depths from ICESat-2 ATL24 HDF5 files.

Outputs
-------
1. CSV with WGS84 longitude/latitude columns.
2. GeoPackage point layer in EPSG:4326.

Depth convention
----------------
depth_m = surface_h - ortho_h

Both ATL24 variables are orthometric heights relative to the geoid. Positive
values therefore represent distance downward from the ATL24-detected local sea
surface to the refraction-corrected seafloor photon.

No interpolation, rasterization, tide reduction, datum transformation, spatial
averaging, or substitution of classification/confidence fields is performed.
Only ATL24 photons whose class_ph equals 40 (bathymetry) are exported.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd

BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
BATHYMETRY_CLASS = 40

# Required for direct depth derivation and geolocation.
REQUIRED = (
    "class_ph",
    "lon_ph",
    "lat_ph",
    "ortho_h",
    "surface_h",
    "delta_time",
)

# Export when available. Missing optional datasets become null columns.
OPTIONAL = (
    "ellipse_h",
    "confidence",
    "sigma_thu",
    "sigma_tvu",
    "x_atc",
    "y_atc",
    "index_ph",
    "index_seg",
    "low_confidence_flag",
    "sensor_depth_exceeded",
    "invalid_kd",
    "invalid_wind_speed",
    "night_flag",
)


def expand_inputs(values: Iterable[str]) -> list[Path]:
    """Expand file paths, directories, and shell-style glob patterns."""
    found: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            matches = sorted(path.glob("*.h5")) + sorted(path.glob("*.hdf5"))
        elif any(char in value for char in "*?[]"):
            matches = [Path(p) for p in sorted(glob.glob(value))]
        else:
            matches = [path]
        found.extend(matches)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise FileNotFoundError("No ATL24 HDF5 files matched the input.")
    missing = [str(p) for p in unique if not p.is_file()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))
    return unique


def read_dataset(group: h5py.Group, name: str, required: bool = False):
    """Read a one-dimensional HDF5 dataset and replace declared fills by NaN."""
    if name not in group:
        if required:
            raise KeyError(f"Missing required dataset {group.name}/{name}")
        return None

    dataset = group[name]
    values = np.asarray(dataset[:])
    if values.ndim != 1:
        raise ValueError(f"Expected 1-D dataset at {dataset.name}, got {values.shape}")

    # Preserve integer flag/index arrays unless they contain a declared fill.
    fill = dataset.attrs.get("_FillValue", dataset.attrs.get("fillvalue", None))
    if fill is not None and np.issubdtype(values.dtype, np.number):
        fill_scalar = np.asarray(fill).reshape(-1)[0]
        hit = values == fill_scalar
        if np.any(hit):
            values = values.astype(float)
            values[hit] = np.nan
    return values


def granule_metadata(h5: h5py.File) -> dict[str, object]:
    """Read selected scalar orbit metadata without inventing missing values."""
    result: dict[str, object] = {}
    paths = {
        "rgt": "orbit_info/rgt",
        "cycle_number": "orbit_info/cycle_number",
        "orbit_number": "orbit_info/orbit_number",
        "sc_orient": "orbit_info/sc_orient",
    }
    for column, h5_path in paths.items():
        if h5_path in h5:
            value = np.asarray(h5[h5_path][()]).reshape(-1)
            result[column] = value[0].item() if len(value) else pd.NA
        else:
            result[column] = pd.NA
    return result


def extract_file(path: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    with h5py.File(path, "r") as h5:
        metadata = granule_metadata(h5)
        present_beams = [beam for beam in BEAMS if beam in h5]
        if not present_beams:
            raise ValueError(f"No ATL24 beam groups found in {path}")

        for beam in present_beams:
            group = h5[beam]
            arrays = {name: read_dataset(group, name, required=True) for name in REQUIRED}
            n = len(arrays["class_ph"])
            if any(len(values) != n for values in arrays.values()):
                raise ValueError(f"Required ATL24 arrays have unequal lengths in {path}:{beam}")

            mask = np.asarray(arrays["class_ph"] == BATHYMETRY_CLASS)
            mask &= np.isfinite(np.asarray(arrays["lon_ph"], dtype=float))
            mask &= np.isfinite(np.asarray(arrays["lat_ph"], dtype=float))
            mask &= np.isfinite(np.asarray(arrays["ortho_h"], dtype=float))
            mask &= np.isfinite(np.asarray(arrays["surface_h"], dtype=float))
            if not np.any(mask):
                continue

            data: dict[str, object] = {
                "source_file": np.repeat(path.name, int(mask.sum())),
                "beam": np.repeat(beam, int(mask.sum())),
                "class_ph": np.asarray(arrays["class_ph"])[mask],
                "longitude": np.asarray(arrays["lon_ph"], dtype=float)[mask],
                "latitude": np.asarray(arrays["lat_ph"], dtype=float)[mask],
                "seafloor_ortho_h_m": np.asarray(arrays["ortho_h"], dtype=float)[mask],
                "surface_ortho_h_m": np.asarray(arrays["surface_h"], dtype=float)[mask],
                "delta_time_s": np.asarray(arrays["delta_time"], dtype=float)[mask],
            }
            data["depth_m"] = data["surface_ortho_h_m"] - data["seafloor_ortho_h_m"]

            for name in OPTIONAL:
                values = read_dataset(group, name, required=False)
                if values is None:
                    data[name] = np.full(int(mask.sum()), np.nan)
                elif len(values) != n:
                    raise ValueError(f"Unequal optional array length for {path}:{beam}/{name}")
                else:
                    data[name] = values[mask]

            for name, value in metadata.items():
                data[name] = np.repeat(value, int(mask.sum()))

            frame = pd.DataFrame(data)
            # Physical consistency check only: a bathymetry depth must be positive.
            # Retain non-positive records, but mark them rather than silently deleting.
            frame["positive_depth"] = frame["depth_m"] > 0.0
            frames.append(frame)
    return frames


def apply_user_filters(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    keep = np.ones(len(frame), dtype=bool)
    if args.min_confidence is not None:
        if frame["confidence"].isna().all():
            raise ValueError("--min-confidence requested, but ATL24 confidence is unavailable.")
        keep &= frame["confidence"].fillna(-np.inf).to_numpy() >= args.min_confidence
    if args.exclude_low_confidence:
        keep &= frame["low_confidence_flag"].fillna(1).to_numpy() == 0
    if args.exclude_sensor_depth_exceeded:
        keep &= frame["sensor_depth_exceeded"].fillna(1).to_numpy() == 0
    if args.positive_depth_only:
        keep &= frame["positive_depth"].to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export ATL24 class-40 bathymetric photon depths to CSV and GeoPackage."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="ATL24 .h5/.hdf5 files, directories, or quoted glob patterns."
    )
    parser.add_argument("--csv", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--gpkg", required=True, type=Path, help="Output GeoPackage path.")
    parser.add_argument("--layer", default="atl24_bathymetry", help="GeoPackage layer name.")
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="Optional ATL24 ensemble-confidence threshold in [0,1]. No threshold by default."
    )
    parser.add_argument(
        "--exclude-low-confidence", action="store_true",
        help="Exclude photons where low_confidence_flag is true or missing."
    )
    parser.add_argument(
        "--exclude-sensor-depth-exceeded", action="store_true",
        help="Exclude photons where sensor_depth_exceeded is true or missing."
    )
    parser.add_argument(
        "--positive-depth-only", action="store_true",
        help="Exclude records whose calculated depth_m is zero or negative."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing CSV and GeoPackage files."
    )
    args = parser.parse_args()

    if args.min_confidence is not None and not (0.0 <= args.min_confidence <= 1.0):
        parser.error("--min-confidence must be between 0 and 1.")

    inputs = expand_inputs(args.inputs)
    for output in (args.csv, args.gpkg):
        output = output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            parser.error(f"Output exists: {output}. Use --overwrite to replace it.")
        output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for path in inputs:
        frames.extend(extract_file(path))
    if not frames:
        raise RuntimeError("No ATL24 class-40 bathymetric photons were found.")

    result = pd.concat(frames, ignore_index=True)
    result = apply_user_filters(result, args)
    if result.empty:
        raise RuntimeError("No bathymetric photons remain after the requested filters.")

    # Deterministic order makes repeated exports straightforward to compare.
    result = result.sort_values(
        ["source_file", "beam", "delta_time_s", "index_ph"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "feature_id", np.arange(1, len(result) + 1, dtype=np.int64))

    csv_path = args.csv.expanduser().resolve()
    gpkg_path = args.gpkg.expanduser().resolve()
    if args.overwrite:
        csv_path.unlink(missing_ok=True)
        gpkg_path.unlink(missing_ok=True)

    result.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.9f")

    geodata = gpd.GeoDataFrame(
        result,
        geometry=gpd.points_from_xy(result["longitude"], result["latitude"]),
        crs="EPSG:4326",
    )
    geodata.to_file(gpkg_path, layer=args.layer, driver="GPKG", index=False)

    print(f"ATL24 input files: {len(inputs)}")
    print(f"Bathymetric points written: {len(result)}")
    print(f"CSV: {csv_path}")
    print(f"GeoPackage: {gpkg_path} (layer: {args.layer})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
