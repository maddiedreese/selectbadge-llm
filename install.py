#!/usr/bin/env python3
"""Install the badge LLM demo into a pimoroni/badgeware-cpp checkout."""
from pathlib import Path
import shutil
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 install.py /path/to/badgeware-cpp")

root = Path(sys.argv[1]).resolve()
demo_src = Path(__file__).resolve().parent / "demos" / "badge_llm"
demo_dst = root / "demos" / "badge_llm"
list_file = root / "demos" / "CMakeLists.txt"
if not list_file.is_file() or not (root / "lib" / "badge.hpp").is_file():
    raise SystemExit("Target is not a badgeware-cpp checkout")
shutil.copytree(demo_src, demo_dst, dirs_exist_ok=True)
text = list_file.read_text()
if "badge_llm" not in text:
    marker = "set(DEMOS\n"
    if marker not in text:
        raise SystemExit("Could not locate the demo list in CMakeLists.txt")
    list_file.write_text(text.replace(marker, marker + "  badge_llm ", 1))
print(f"Installed badge_llm into {root}")
