import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablate import paired

print("paired, negative is the first better:")
for a, b in (("fit_b75", "fit_epa_weekly"), ("fit_sr", "fit_epa_weekly"), ("fit_b75_still", "fit_epa"),
             ("fit_epa_weekly", "fit_epa"), ("fit_b75", "fit_b75_still"), ("fit_b75", "base")):
    paired(a, b)
