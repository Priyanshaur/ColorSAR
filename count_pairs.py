"""Count the s1/s2 files and matched pairs per terrain (uses the same pairing as training)."""
import os
import sys
from collections import Counter

from color_utils import find_pairs

root = sys.argv[1] if len(sys.argv) > 1 else "v_2"
pairs = find_pairs(root)
per_terrain = Counter(t for _, _, t in pairs)

print(f"{'terrain':<12}{'s1':>7}{'s2':>7}{'pairs':>7}")
for terrain in sorted(os.listdir(root)) if os.path.isdir(root) else []:
    n = {s: len([f for f in os.listdir(os.path.join(root, terrain, s)) if f.endswith(".png")])
         if os.path.isdir(os.path.join(root, terrain, s)) else 0 for s in ("s1", "s2")}
    print(f"{terrain:<12}{n['s1']:>7}{n['s2']:>7}{per_terrain[terrain]:>7}")
print(f"{'total':<12}{'':>14}{len(pairs):>7}")
