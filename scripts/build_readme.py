"""Render the complete repository homepage from the committed resource tables."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dtd", "eurosat", "oxford_pets", "sun397")
DATASET_NAMES = ("DTD", "EuroSAT", "Pets", "SUN397")
ENCODERS = {"clip": "CLIP ViT-B/16", "siglip2": "SigLIP 2 Base/16"}
METHODS = [
    ("lp", "LP", "Base"), ("lpplusplus", "LP", "LP++"), ("proker", "ProKeR", "Kernel solve"),
    ("prolip", "ProLIP", "Full"), ("lora_rank", "LoRA", "Rank"), ("lora_param", "LoRA", "Param."),
    ("svd", "SVD-E", "Singular values"), ("comp_k2", "Comp-E", "k = 2"),
    ("comp_rank", "Comp-E", "Rank"), ("comp_param", "Comp-E", "Param."), ("p0", "P0", "Task-text"),
]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def table(headers, rows, text_columns=2):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * text_columns + ["---:"] * (len(headers)-text_columns)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def metric(row, signed=False):
    if row is None or row.get("mean") is None:
        return "—"
    return (f"{row['mean']:+.3f} ± {row['sample_sd']:.3f}" if signed else
            f"{row['mean']:.2f} ± {row['sample_sd']:.2f}")


def render():
    budgets, results = read("data/parameter_budgets.json"), read("data/paper_results.json")
    counts = {(r["family"], r["dataset"], r["method"]): r for r in budgets["cells"]}
    generated = {}
    blocks = []
    for family, encoder in ENCODERS.items():
        rows, previous = [], None
        for method, title, variant in METHODS:
            label = "" if title == previous else "**P0**" if title == "P0" else title
            values = [counts[family, ds, method]["parameters"] for ds in DATASETS]
            rows.append([label, "*" + variant + "*"] + ["Closed-form" if v is None else f"{v:,}" for v in values])
            previous = title
        blocks.append("### " + encoder + "\n\n" + table(["Method", "Configuration", *DATASET_NAMES], rows))
    generated["parameters"] = "\n\n".join(blocks)
    generated["proker_counts"] = table(["Support images/class", *DATASET_NAMES],
        [[k] + [f"{10*k*c*c:,}" for c in (47, 10, 37, 397)] for k in (4, 16)], text_columns=1)
    rows = []
    for family in ENCODERS:
        for ds, title in zip(DATASETS, DATASET_NAMES):
            p0, lo, co = (counts[family, ds, method] for method in ("p0", "lora_param", "comp_param"))
            rows.append([ENCODERS[family], title, p0["rank"], lo["rank"],
                         f"{lo['parameters']-p0['parameters']:+,}", co["rank"],
                         f"{co['parameters']-p0['parameters']:+,}"])
    generated["matched"] = table(["Encoder", "Dataset", "P0 / Rank r", "LoRA Param. k",
                                  "LoRA difference", "Comp-E Param. k", "Comp-E difference"], rows)
    for key, signed in (("main", False), ("paired", True)):
        lookup = {(x["family"], x["dataset"], x["shot"], x["method"]): x for x in results[key]}
        blocks = []
        for family, encoder in ENCODERS.items():
            winners = {}
            if not signed:
                for group in ("LP", "LoRA", "Comp-E"):
                    choices = [m for m, title, _ in METHODS if title == group]
                    winners[group] = max(choices, key=lambda m: statistics.mean(
                        lookup[family, ds, k, m]["mean"] for ds in DATASETS for k in (4, 16)))
            rows, previous = [], None
            for method, title, variant in METHODS:
                if signed and method == "p0":
                    continue
                label = "" if title == previous else title
                if not signed and winners.get(title) == method:
                    variant = "**" + variant + "**"
                rows.append([label, "*" + variant + "*"] + [
                    metric(lookup.get((family, ds, k, method)), signed)
                    for ds in DATASETS for k in (4, 16)])
                previous = title
            blocks.append("### " + encoder + "\n\n" + table(
                ["Method", "Configuration"] + [f"{ds} {k}" for ds in DATASET_NAMES for k in (4, 16)], rows))
        generated["paired" if signed else "accuracy"] = "\n\n".join(blocks)

    specs = [
        ("selected_coordinates", "selected_coordinates", ["p0", "r0", "r1", "r2"], ["P0", "R0", "R1", "R2"]),
        ("fixed_coordinates", "fixed_coordinates", ["p0", "r0", "r1", "r2"], ["P0", "R0", "R1", "R2"]),
        ("normalization", "normalization_regularization",
         ["p0", "dynamic_ci", "fixed_ci", "fixed_disp", "r0"],
         ["P0: D/Disp.", "D/CI", "F/CI", "F/Disp.", "Random"]),
        ("position", "position", ["p0_output", "prolip_output", "p0_fc2", "prolip_fc2"],
         ["P0 output", "ProLIP output", "P0 FC2", "ProLIP FC2"]),
    ]
    ds_names = dict(zip(DATASETS, DATASET_NAMES))
    for label, source, keys, headers in specs:
        generated[label] = table(["Encoder", "Dataset / shot", *headers], [
            [ENCODERS[row["family"]], f"{ds_names[row['dataset']]} / {row['shot']}"] +
            [metric(row["methods"][key]) for key in keys] for row in results[source]])
    generated["numerics"] = table(["Encoder / task", "FP32 solve relative residual",
                                    "FP64 logit max error", "FP64 loss error", "FP64 gradient max error"], [
        [ENCODERS[row["family"]] + " / " + ds_names[row["dataset"]]] + [
            f"{row[k]:.3e}" for k in ("solve_relative_residual_fp32",
                "fixed_denominator_logits_fp64_max_abs", "fixed_denominator_loss_fp64_abs",
                "fixed_denominator_gradient_fp64_max_abs")] for row in results["coordinate_numerics"]],
                text_columns=1)
    text = (ROOT / "docs/README.template.md").read_text(encoding="utf-8")
    for name, content in generated.items():
        marker = "<!-- TABLE:" + name + " -->"
        assert text.count(marker) == 1, marker
        text = text.replace(marker, content)
    assert "<!-- TABLE:" not in text
    return text


if __name__ == "__main__":
    text = render()
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print(f"Rendered README: {len(text.splitlines())} lines")
