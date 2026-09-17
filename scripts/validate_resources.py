"""Check arithmetic, seed pairing, README freshness and public file references."""
import json
import math
from pathlib import Path
import re
import statistics
from build_readme import render

ROOT = Path(__file__).resolve().parents[1]


def main():
    load = lambda name: json.loads((ROOT / name).read_text(encoding="utf-8"))
    data = load("data/paper_results.json")
    values = {(r["family"], r["dataset"], r["shot"], r["seed"], r["method"]): r["accuracy"]
              for r in data["per_seed"]}
    assert len(values) == len(data["per_seed"]) == 522
    checked = 0
    for row in data["main"]:
        if row["mean"] is None:
            assert row["method"] == "proker" and row["dataset"] == "sun397" and row["shot"] == 16
            continue
        seed_values = [values[row["family"], row["dataset"], row["shot"], s, row["method"]] for s in (1, 2, 3)]
        assert math.isclose(statistics.mean(seed_values), row["mean"], abs_tol=1e-9)
        assert math.isclose(statistics.stdev(seed_values), row["sample_sd"], abs_tol=1e-9)
        checked += 1
    assert checked == 174
    for key in ("selected_coordinates", "fixed_coordinates", "normalization_regularization", "position"):
        for row in data[key]:
            for method, cell in row["methods"].items():
                assert len(cell["values"]) == 3
                assert math.isclose(statistics.mean(cell["values"]), cell["mean"], abs_tol=1e-9)
                assert math.isclose(statistics.stdev(cell["values"]), cell["sample_sd"], abs_tol=1e-9)
                if key == "selected_coordinates" and method == "p0":
                    assert cell["values"] == [values[row["family"], row["dataset"], row["shot"], s, "p0"]
                                              for s in (1, 2, 3)]
    for row in data["paired"]:
        expected = [values[row["family"], row["dataset"], row["shot"], s, "p0"] -
                    values[row["family"], row["dataset"], row["shot"], s, row["method"]] for s in (1, 2, 3)]
        assert expected == row["values"]
        assert math.isclose(statistics.mean(expected), row["mean"], abs_tol=1e-9)
        assert math.isclose(statistics.stdev(expected), row["sample_sd"], abs_tol=1e-9)
    counts = 0
    for row in load("data/parameter_budgets.json")["cells"]:
        d, e = 768, (512 if row["family"] == "clip" else 768)
        dc = 768 if row["family"] == "clip" else 3072
        c, r, k, method = row["classes"], row["text_rank"], row["rank"], row["method"]
        expected = {"lp": c*(e+1), "lpplusplus": c*(e+2), "proker": None,
                    "prolip": d*e, "p0": d*r, "svd": min(dc, e)}.get(method)
        if method.startswith("lora"):
            expected = k*(d+e)
        if method.startswith("comp"):
            expected = k*(dc+e-32)
        assert expected == row["parameters"]
        if method in ("lora_param", "comp_param"):
            unit = d+e if method.startswith("lora") else dc+e-32
            assert k == min(range(1, e+1), key=lambda n: (abs(n*unit-d*r), n))
        counts += expected is not None
    assert counts == 80
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    assert content == render(), "Run scripts/build_readme.py"
    for target in re.findall(r"\]\(([^)]+)\)", content):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (ROOT / target.split("#")[0]).exists(), target
    assert not re.search(r"\b(added|reused|extension|expanded|five.seed|post.hoc)\b", content, re.I)
    for file in [ROOT / "README.md", *ROOT.glob("data/**/*.json"), *ROOT.glob("configs/*.json"),
                 *ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]:
        text = file.read_text(encoding="utf-8")
        assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text), file
    print(json.dumps({"main_numeric_cells": checked, "per_seed_accuracies": 522, "parameter_counts": counts,
                      "paired_cells": len(data["paired"]), "readme_current": True}))


if __name__ == "__main__":
    main()
