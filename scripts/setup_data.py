from __future__ import annotations
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterator, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ask_the_sensors.config import load_config  
from ask_the_sensors.cv_split import load_cv_folds, summarize_folds 
from ask_the_sensors.data_io import list_available_uuids, UUID_RE 

PRIMARY_FILE_RE = re.compile(
    r"^([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"
    r"\.features_labels\.csv(\.gz)?$"
)
FOLD_FILE_RE = re.compile(r"^fold_\d+_(train|test)_(iphone|android)_uuids\.txt$")

def _iter_zip_entries(zip_path: Path) -> Iterator[Tuple[Path, str, zipfile.ZipInfo]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            basename = Path(member.filename).name
            if not basename:
                continue
            yield zip_path, basename, member

def _copy_from_zip(zip_path: Path, member: zipfile.ZipInfo, dest_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(member) as src, open(dest_path, "wb") as dst:
        dst.write(src.read())

def collect_dataset_files(downloads_dir: Path, primary_dest: Path, cv_dest: Path) -> Tuple[int, int]:
    primary_dest.mkdir(parents=True, exist_ok=True)
    cv_dest.mkdir(parents=True, exist_ok=True)

    n_primary = 0
    n_cv = 0

    def _maybe_write(dest_dir: Path, basename: str, read_bytes) -> bool:
        target = dest_dir / basename
        if target.is_file() and target.stat().st_size > 0:
            return False  
        data = read_bytes()
        with open(target, "wb") as f:
            f.write(data)
        return True

    for path in downloads_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if PRIMARY_FILE_RE.match(name):
            if _maybe_write(primary_dest, name, path.read_bytes):
                n_primary += 1
        elif FOLD_FILE_RE.match(name):
            if _maybe_write(cv_dest, name, path.read_bytes):
                n_cv += 1

    for zip_path in downloads_dir.rglob("*.zip"):
        try:
            for zpath, basename, member in _iter_zip_entries(zip_path):
                if PRIMARY_FILE_RE.match(basename):
                    if _maybe_write(primary_dest, basename, lambda m=member, z=zpath: _read_zip_member(z, m)):
                        n_primary += 1
                elif FOLD_FILE_RE.match(basename):
                    if _maybe_write(cv_dest, basename, lambda m=member, z=zpath: _read_zip_member(z, m)):
                        n_cv += 1
        except zipfile.BadZipFile:
            print(f"  WARNING: could not open {zip_path} (corrupt or incomplete download?); skipping.")

    return n_primary, n_cv

def _read_zip_member(zip_path: Path, member: zipfile.ZipInfo) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(member)

def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()

    downloads_dir = cfg.downloads_dir
    if not downloads_dir.is_dir() or not any(downloads_dir.iterdir()):
        print(f"ERROR: {downloads_dir} is empty.\n")
        print("  Place your downloaded/extracted ExtraSensory data here. Accepted forms:")
        print(f"    - the bulk zip:      {cfg.raw['download_urls']['primary_data_zip']}")
        print(f"    - per-user zips, e.g. <UUID>_features_labels_csv.zip (one per user)")
        print(f"    - a cross-validation partition zip, e.g.:")
        print(f"        {cfg.raw['download_urls']['cv_folds_zip']}")
        print(f"    - or any of the above already extracted, in any subfolder layout")
        print("See README.md section 2 for full instructions.")
        return 1

    already_have = list_available_uuids(cfg)
    cv_files_present = len(list(cfg.cv_folds_dir.glob("fold_*_*.txt")))

    if len(already_have) >= 60 and cv_files_present >= 20:
        print(
            f"Primary data ({len(already_have)} users) and CV folds ({cv_files_present} files) "
            f"already present; skipping collection."
        )
    else:
        print(f"Scanning {downloads_dir} recursively for dataset files (this may take a moment "
              f"if it contains large zip archives)...")
        n_primary, n_cv = collect_dataset_files(downloads_dir, cfg.primary_data_dir, cfg.cv_folds_dir)
        print(f"  copied {n_primary} new primary-data file(s), {n_cv} new CV fold file(s)")

    # ---- Validate ------------------------------------------------------------
    uuids = list_available_uuids(cfg)
    print(f"\nFound {len(uuids)} user primary-data files in {cfg.primary_data_dir}")
    if len(uuids) != 60:
        print(
            f"WARNING: expected 60 users, found {len(uuids)}. "
            f"If you're still collecting per-user files this is expected; otherwise, check that "
            f"every user's file/zip is somewhere under {downloads_dir}."
        )

    if not uuids:
        print("ERROR: no usable primary-data files found. Aborting.")
        return 1

    try:
        folds = load_cv_folds(cfg)
        print("\nCross-validation partition:")
        print(summarize_folds(folds))
    except FileNotFoundError as e:
        print(f"ERROR validating CV folds: {e}")
        return 1

    print("\nSetup complete. You can now run: python -m scripts.run_all")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())