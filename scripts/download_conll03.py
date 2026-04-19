#!/usr/bin/env python3
"""Download CoNLL-2003 English raw files into data/raw/.

The legacy HuggingFace ``conll2003`` dataset is script-based and no longer
loadable on ``datasets`` >= 3.0. This script instead fetches the canonical
four-column files directly from a well-known public mirror:

    https://github.com/synalp/NER  (corpus/CoNLL-2003/)

Outputs three files in the original 4-column ``token POS chunk NER``
layout (IOB1 tagging scheme, matching what ``src/data_utils.parse_conll03_file``
and the CleanCoNLL build script expect):

    data/raw/eng.train
    data/raw/eng.testa
    data/raw/eng.testb

Neither the HF mirror nor this GitHub mirror is an official Reuters
distribution. If your use case requires the original license, obtain the
data through the official CoNLL-2003 shared task channels instead and skip
this script.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request


DEFAULT_MIRROR = (
    "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003"
)

# Remote filename -> local filename the pipeline expects.
FILES = {
    "eng.train": "eng.train",
    "eng.testa": "eng.testa",
    "eng.testb": "eng.testb",
}


def _download(url: str, dest: str) -> int:
    """Download ``url`` to ``dest``. Returns bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        data = resp.read()
        out.write(data)
    return len(data)


def _looks_like_conll(path: str) -> bool:
    """Heuristic: first non-empty line should have 4 space-separated columns."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                return len(parts) == 4
    except OSError:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to write the three raw files (default: data/raw)",
    )
    parser.add_argument(
        "--mirror",
        default=DEFAULT_MIRROR,
        help=f"Base URL serving eng.train/testa/testb (default: {DEFAULT_MIRROR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    any_written = False
    for remote_name, local_name in FILES.items():
        dest = os.path.join(args.output_dir, local_name)
        if os.path.exists(dest) and not args.force:
            print(f"  Skipping {dest}: already exists (use --force to overwrite).")
            continue

        url = f"{args.mirror}/{remote_name}"
        print(f"  Downloading {url}")
        try:
            n = _download(url, dest)
        except urllib.error.HTTPError as exc:
            sys.exit(f"  HTTP {exc.code} fetching {url}: {exc.reason}")
        except urllib.error.URLError as exc:
            sys.exit(f"  Network error fetching {url}: {exc.reason}")

        if not _looks_like_conll(dest):
            sys.exit(
                f"  {dest} does not look like CoNLL format "
                f"(expected 4 space-separated columns on the first data line). "
                f"Check the --mirror URL."
            )

        print(f"    wrote {dest} ({n:,} bytes)")
        any_written = True

    if any_written:
        print(
            "\nDone. Next: `python scripts/prepare_data.py --config configs/default.yaml`"
        )
    else:
        print("\nNothing to do.")


if __name__ == "__main__":
    main()
