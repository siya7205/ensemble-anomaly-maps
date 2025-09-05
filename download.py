# download.py — robust Dataverse downloader (resume + 416 handling)
import os, time, requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DOI = "doi:10.17617/3.8O"
API = "https://edmond.mpg.de/api"
OUT = Path("data/raw_trajectory")

# what to pull for the demo
WANTED = {"align_topol.pdb", "trajectory_0.xtc"}

def session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1.0,
              status_forcelist=[429,502,503,504],
              allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s

def list_files(s):
    r = s.get(f"{API}/datasets/:persistentId/",
              params={"persistentId": DOI}, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for f in data["data"]["latestVersion"]["files"]:
        out.append({
            "id": f["dataFile"]["id"],
            "label": f["label"],
            "size": f["dataFile"].get("filesize", 0),
        })
    return out

def download_file(s, file_id, label, dest, expected_size):
    dest.parent.mkdir(parents=True, exist_ok=True)

    # If file already complete, skip
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"[skip] {label} (already complete {expected_size/1e6:.1f} MB)")
        return

    base_url = f"{API}/access/datafile/{file_id}"
    headers = {}

    # Try resume if partial exists
    if dest.exists() and dest.stat().st_size < (expected_size or 1<<60):
        headers["Range"] = f"bytes={dest.stat().st_size}-"

    def _stream_to_file(resp, mode):
        with open(dest, mode) as f:
            got = dest.stat().st_size if mode == "ab" else 0
            total = int(resp.headers.get("Content-Length", 0)) + (got if mode=="ab" else 0)
            for chunk in resp.iter_content(chunk_size=256*1024):
                if not chunk: continue
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r[dl ] {label}: {got/1e6:6.1f}/{total/1e6:6.1f} MB", end="")
            print()

    # First attempt (maybe resume)
    r = s.get(base_url, params={"format":"original"}, headers=headers, stream=True, timeout=60)
    if r.status_code == 416:
        # 416: our Range is invalid (file complete or too large local size)
        # fall back to clean download from zero
        print(f"[warn] 416 on {label} — restarting from zero")
        if dest.exists():
            dest.unlink()
        r = s.get(base_url, params={"format":"original"}, stream=True, timeout=60)

    r.raise_for_status()
    mode = "ab" if headers.get("Range") and r.status_code == 206 else "wb"
    print(f"[get ] {label} -> {dest}")
    _stream_to_file(r, mode)
    print(f"[ok  ] {label}")

def main():
    s = session()
    print(f"[info] listing dataset files…")
    files = list_files(s)
    want = {f["label"]: f for f in files if f["label"] in WANTED}
    if not want:
        print("[error] none of the requested files were found")
        return
    for label, meta in want.items():
        dest = OUT / label
        download_file(s, meta["id"], label, dest, meta.get("size", 0))

if __name__ == "__main__":
    main()
