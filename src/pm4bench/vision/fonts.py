"""Download the pinned Noto fonts and their SIL OFL notices, outside the dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from ..data.benchmark import sha256
from .resources import RESOURCE_ROOT


def download_fonts(output: Path) -> None:
    sources = json.loads((RESOURCE_ROOT / "fonts.lock.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        for filename, url, digest in (
            (name, source["url"], source["sha256"]),
            (name + ".OFL.txt", source["license_url"], source["license_sha256"]),
        ):
            path = output / filename
            if path.exists():
                if sha256(path) != digest:
                    raise FileExistsError(f"Existing {filename} has different contents")
                continue
            with urllib.request.urlopen(url, timeout=600) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch downloading {filename}")
            with path.open("xb") as handle:
                handle.write(data)
        print(f"Verified {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    download_fonts(parser.parse_args().output)
