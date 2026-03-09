from __future__ import annotations

from pathlib import Path
import h5py
import pandas as pd
from IPython.display import Markdown, display

EXPECTED_KEYS = {"ismrmrd_header", "reconstruction_rss"}


def bytes_to_gib(n: int) -> float:
    return n / (1024 ** 3)


def folder_size_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*.h5"):
        total += p.stat().st_size
    return total


def list_h5_relative(root: Path) -> list[Path]:
    return sorted([p.relative_to(root) for p in root.rglob("*.h5")])


def validate_candidate(reference_root: Path, my_root: Path, expected_keys: set[str] | None = None):
    expected = expected_keys if expected_keys is not None else EXPECTED_KEYS

    if not reference_root.exists():
        raise FileNotFoundError(f"Reference root not found: {reference_root}")
    if not my_root.exists():
        raise FileNotFoundError(f"Candidate root not found: {my_root}")

    ref_files = list_h5_relative(reference_root)
    my_files = list_h5_relative(my_root)

    ref_set = set(ref_files)
    my_set = set(my_files)
    missing = sorted(ref_set - my_set)
    extra = sorted(my_set - ref_set)
    shared = sorted(ref_set & my_set)

    rows = []
    for rel in shared:
        rp = reference_root / rel
        mp = my_root / rel
        try:
            with h5py.File(rp, "r") as rf, h5py.File(mp, "r") as mf:
                mkeys = set(mf.keys())
                keys_ok = mkeys == expected
                has_kspace = "kspace" in mkeys
                ref_shape = tuple(rf["reconstruction_rss"].shape) if "reconstruction_rss" in rf else None
                my_shape = tuple(mf["reconstruction_rss"].shape) if "reconstruction_rss" in mf else None
                shape_ok = ref_shape == my_shape
                rows.append(
                    {
                        "relative_path": str(rel),
                        "keys_ok": keys_ok,
                        "shape_ok": shape_ok,
                        "has_kspace": has_kspace,
                        "my_keys": sorted(list(mkeys)),
                        "error": None,
                    }
                )
        except Exception as e:
            rows.append(
                {
                    "relative_path": str(rel),
                    "keys_ok": False,
                    "shape_ok": False,
                    "has_kspace": False,
                    "my_keys": None,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows)
    shared_n = len(shared)
    size_b = folder_size_bytes(my_root)
    size_gib = bytes_to_gib(size_b)

    summary = {
        "reference_file_count": len(ref_files),
        "candidate_file_count": len(my_files),
        "shared_file_count": shared_n,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "name_match_100": (len(missing) == 0 and len(extra) == 0),
        "keys_ok_count": int(df["keys_ok"].sum()) if shared_n else 0,
        "shape_ok_count": int(df["shape_ok"].sum()) if shared_n else 0,
        "kspace_present_count": int(df["has_kspace"].sum()) if shared_n else 0,
        "error_count": int(df["error"].notna().sum()) if shared_n else 0,
        "size_bytes": size_b,
        "size_gib": size_gib,
    }

    return {"summary": summary, "df": df, "missing": missing, "extra": extra}


def run_validation_report(reference_root: Path, my_root: Path, max_folder_size_gib: float = 8.0, expected_keys: set[str] | None = None):
    expected = expected_keys if expected_keys is not None else EXPECTED_KEYS
    report = validate_candidate(reference_root=reference_root, my_root=my_root, expected_keys=expected)
    s = report["summary"]

    shared = max(1, s["shared_file_count"])

    name_ok = bool(s["name_match_100"])
    keys_ok = s["keys_ok_count"] == s["shared_file_count"] and s["shared_file_count"] > 0
    shape_ok = s["shape_ok_count"] == s["shared_file_count"] and s["shared_file_count"] > 0
    kspace_ok = s["kspace_present_count"] == 0
    error_ok = s["error_count"] == 0
    size_ok = s["size_gib"] <= max_folder_size_gib

    ready_to_package = name_ok and keys_ok and shape_ok and kspace_ok and error_ok and size_ok

    print("=" * 96)
    print("Validation Report")
    print("=" * 96)
    print(f"Reference folder: {reference_root}")
    print(f"Candidate folder: {my_root}")
    print(f"Expected candidate keys: {sorted(list(expected))}")
    print()
    print("File structure")
    print(f"- Reference files: {s['reference_file_count']}")
    print(f"- Candidate files: {s['candidate_file_count']}")
    print(f"- Shared files:    {s['shared_file_count']}")
    print(f"- Missing files in candidate: {s['missing_count']}")
    print(f"- Extra files in candidate:   {s['extra_count']}")
    print(f"- Name/structure 100% match:  {name_ok}")
    print()
    print("Content checks on shared files")
    print(f"- Keys exactly expected:      {s['keys_ok_count']} / {shared}")
    print(f"- reconstruction_rss shapes:  {s['shape_ok_count']} / {shared}")
    print(f"- Files that still contain kspace: {s['kspace_present_count']}")
    print(f"- K-space absent in all files: {kspace_ok}")
    print(f"- Read/parse errors:          {s['error_count']}")
    print()
    print("Size check")
    print(f"- Candidate size: {s['size_gib']:.3f} GiB")
    print(f"- Max allowed:    {max_folder_size_gib:.3f} GiB")
    print(f"- Size within limit: {size_ok}")
    print("=" * 96)

    if ready_to_package:
        display(Markdown("### Ready for Submission"))
        print("All checks passed: names, keys, shapes, and size.")
    else:
        display(Markdown("### Not ready yet"))
        print("One or more checks failed. Review details below.")
        if s["kspace_present_count"] > 0:
            display(Markdown("### K-space detected"))
            print("Some candidate files still contain kspace.")
            print("Use `export_reconstruction_only_folder(...)` to create a clean folder without kspace.")

    if s["missing_count"] > 0:
        print()
        print("Examples missing in candidate:")
        for p in report["missing"][:12]:
            print("-", p)

    if s["extra_count"] > 0:
        print()
        print("Examples extra in candidate:")
        for p in report["extra"][:12]:
            print("-", p)

    issues = report["df"][(~report["df"]["keys_ok"]) | (~report["df"]["shape_ok"]) | (report["df"]["error"].notna())]
    if len(issues):
        print()
        print("Issue samples:")
        display(issues[["relative_path", "keys_ok", "shape_ok", "my_keys", "error"]].head(20))

    return ready_to_package, report


def export_reconstruction_only_folder(source_root: Path, output_root: Path, overwrite: bool = False):
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output root already exists and is not empty: {output_root}. "
            "Use overwrite=True or choose another folder."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for src in sorted(source_root.rglob("*.h5")):
        rel = src.relative_to(source_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
            for k, v in fin.attrs.items():
                fout.attrs[k] = v

            for key in ("ismrmrd_header", "reconstruction_rss"):
                if key not in fin:
                    raise KeyError(f"Missing required key '{key}' in {src}")
                fin.copy(fin[key], fout, name=key)

        written += 1

    return {"written_files": written, "output_root": str(output_root)}
