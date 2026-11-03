"""Download the public pi05_libero checkpoint over HTTPS, without GCP tooling.

Run from the repository root. Completed files with matching sizes are reused;
interrupted files are downloaded again. This does not resume individual files.
"""

import concurrent.futures
import json
import pathlib
import shutil
import time
import urllib.parse
import urllib.request

PREFIX = "checkpoints/pi05_libero/"
DESTINATION = pathlib.Path(PREFIX)


def main() -> None:
    listing_url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode(
        {"prefix": PREFIX, "fields": "items(name,size),nextPageToken"}
    )
    with urllib.request.urlopen(listing_url, timeout=60) as response:
        listing = json.load(response)
    if listing.get("nextPageToken"):
        raise RuntimeError("Checkpoint listing requires pagination; refusing an incomplete download")
    items = listing.get("items", [])
    if not items:
        raise RuntimeError("No checkpoint objects found")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "remote_manifest.json").write_text(json.dumps(listing, indent=2))

    def fetch(item: dict) -> str:
        name = item["name"]
        expected = int(item["size"])
        relative = pathlib.PurePosixPath(name.removeprefix(PREFIX))
        if not name.startswith(PREFIX) or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unexpected checkpoint object: {name}")
        destination = DESTINATION / relative
        if destination.is_file() and destination.stat().st_size == expected:
            return f"{destination}: cached"
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        started = time.monotonic()
        for attempt in range(3):
            try:
                url = "https://storage.googleapis.com/openpi-assets/" + urllib.parse.quote(name)
                with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                if partial.stat().st_size != expected:
                    raise ValueError(f"Size mismatch for {destination}")
                partial.replace(destination)
                return f"{destination}: {expected} bytes in {time.monotonic() - started:.1f}s"
            except Exception:
                if attempt == 2:
                    raise
        raise AssertionError("Unreachable")

    print(f"Checking/downloading {len(items)} objects ({sum(int(x['size']) for x in items):,} bytes)", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, items):
            print(result, flush=True)
    (DESTINATION / "DOWNLOAD_COMPLETE").write_text("All remote object sizes verified.\n")
    print(f"Complete: {DESTINATION.resolve()}", flush=True)


if __name__ == "__main__":
    main()
