#!/usr/bin/env python3
"""
Download files from an Edmond (Dataverse) dataset by DOI.

Example:
  python tools/fetch_dataverse.py \
    --doi 10.17617/3.8O \
    --include "traj*.xtc" --include "*.pdb" --include "*.gro" \
    --out data/raw_trajectory
"""
import argparse, fnmatch, os, sys
from pathlib import Path
import requests
from tqdm import tqdm

API_BASE = "https://edmond.mpg.de/api"

def list_files(doi: str):
    # Dataverse: list files in the latest version of a dataset by persistentId
    url = f"{API_BASE}/datasets/:persistentId/versions/:latest?persistentId=doi:{doi}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    files = js["data"]["files"]  # list
    out = []
    for f in files:
        df = f["dataFile"]
        out.append({
            "id": df["id"],
            "filename": df["filename"],
            "contentType": df.get("contentType", ""),
            "size": df.get("filesize", 0),
        })
    return out

def want(filename: str, patterns):
    if not patterns:
        return True
    for pat in patterns:
        if fnmatch.fnmatch(filename, pat):
            return True
    return False

def download_file(file_id: int, dest_path: Path):
    # Dataverse file download by numeric id
    url = f"{API_BASE}/access/datafile/{file_id}?format=original"
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code == 403:
            raise SystemExit(
                "403 Forbidden: this dataset may require you to accept terms in the browser once.\n"
                "Open the dataset page, accept terms (if prompted), then retry."
            )
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as bar:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*256):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi", required=True, help="Dataset DOI, e.g. 10.17617/3.8O")
    ap.add_argument("--include", action="append", default=[],
                    help="Glob(s) to include (repeatable). If none, downloads everything.")
    ap.add_argument("--out", default="data/raw_trajectory", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Listing files in DOI: {args.doi}")
    files = list_files(args.doi)
    if not files:
        print("No files returned. Check DOI.")
        sys.exit(1)

    print(f"[info] Found {len(files)} files total.")
    to_get = [f for f in files if want(f["filename"], args.include)]
    print(f"[info] Will download {len(to_get)} file(s):")
    for f in to_get:
        print(f"  - {f['filename']}  ({f['size']} bytes)")

    for f in to_get:
        dest = out_dir / f["filename"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"[skip] {dest.name} (already present and size matches)")
            continue
        print(f"[get ] {dest.name}")
        download_file(f["id"], dest)

    print(f"\n[done] Files placed in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
