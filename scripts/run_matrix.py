"""Run the complete paper task matrix in separate encoding/training/evaluation phases."""
import argparse
import json
from pathlib import Path
from tasktext.cli import configure, train_cache, evaluate_cache
from tasktext.data import read_json, write_json, load_tensor, encode_cache

ROOT = Path(__file__).resolve().parents[1]


def jobs(config, work, families=None, datasets=None):
    result = []
    for family in config["encoders"]:
        if families and family not in families:
            continue
        for dataset in config["datasets"]:
            if datasets and dataset not in datasets:
                continue
            controls = (family, dataset) in (("clip", "dtd"), ("siglip2", "eurosat"))
            for shots in config["shots"]:
                for seed in config["support_seeds"]:
                    episode = Path(work) / family / dataset / f"k{shots}_s{seed}"
                    specs = [(m, "main", None, False) for m in config["methods"]
                             if not (m == "proker" and dataset == "sun397" and shots == 16)]
                    if controls:
                        specs += [(m, "selected_coordinates", None, False) for m in ("r0", "r1", "r2")]
                        specs += [(m, "fixed_coordinates", None, True) for m in ("p0", "r0", "r1", "r2")]
                        specs += [(m, "normalization", None, False) for m in ("dynamic_ci", "fixed_ci", "fixed_disp")]
                    if family == "siglip2" and dataset in ("dtd", "eurosat"):
                        specs += [(m, "position", "fc2", False) for m in ("p0", "prolip")]
                    for method, group, location, fixed in specs:
                        same = method == "comp_param" and family == "siglip2" and dataset == "eurosat"
                        fitting_method = "comp_k2" if same else method
                        stem = fitting_method + ("_" + location if location else "") + ("_fixed_config" if fixed else "")
                        result.append({"family": family, "dataset": dataset, "shots": shots, "seed": seed,
                            "method": method, "fitting_method": fitting_method, "group": group,
                            "location": location, "fixed": fixed, "cache": str(episode / "cache"),
                            "output": str(episode / group), "model": str(episode / group / (stem + ".pt")),
                            "result": str(episode / group / (stem + "-test.json"))})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-roots")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--phase", choices=("plan", "encode", "train", "evaluate"), required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--families", nargs="+", choices=("clip", "siglip2"))
    p.add_argument("--datasets", nargs="+", choices=("dtd", "eurosat", "oxford_pets", "sun397"))
    p.add_argument("--clip-checkpoint")
    p.add_argument("--siglip2-checkpoint")
    args = p.parse_args()
    config = read_json(ROOT / "configs/paper.json")
    plan = jobs(config, args.work_dir, args.families, args.datasets)
    write_json(Path(args.work_dir) / "task_matrix.json", plan)
    if args.phase == "plan":
        print(json.dumps({"table_entries": len(plan), "fits": len({j["model"] for j in plan}),
                          "episodes": len({j["cache"] for j in plan})}))
        return
    configure(args.device, args.threads)
    roots = read_json(args.data_roots) if args.data_roots else {}
    if args.phase in ("encode", "evaluate") and not all(j["dataset"] in roots for j in plan):
        raise ValueError("--data-roots must specify each selected dataset")
    if args.phase == "evaluate" and not all(Path(j["model"]).exists() for j in plan):
        raise ValueError("Complete selection for this task matrix before evaluation")
    seen = set()
    for job in plan:
        if args.phase in ("encode", "evaluate") and job["cache"] not in seen:
            encode_cache(ROOT / "data/splits" / (job["dataset"] + ".json"), roots[job["dataset"]],
                job["cache"], job["family"], job["shots"], job["seed"], device=args.device,
                checkpoint=args.clip_checkpoint if job["family"] == "clip" else args.siglip2_checkpoint,
                download_root=str(Path(args.work_dir) / "model-cache"), batch_size=args.batch_size,
                splits=("test",) if args.phase == "evaluate" else ("train", "val"))
            seen.add(job["cache"])
        if args.phase == "train" and job["model"] not in seen:
            if Path(job["model"]).exists():
                saved = load_tensor(job["model"])
                base = load_tensor(Path(job["cache"]) / "base.pt")
                if saved.get("selection_complete") is not True or saved["cache_identity"] != base["identity"]:
                    raise ValueError(f"Invalid existing model: {job['model']}")
            else:
                train_cache(job["cache"], job["output"], [job["fitting_method"]], args.device,
                            location=job["location"], fixed=job["fixed"])
            seen.add(job["model"])
        if args.phase == "evaluate" and job["result"] not in seen:
            if not Path(job["result"]).exists():
                evaluate_cache(job["cache"], job["model"], job["result"], args.device, args.batch_size)
            seen.add(job["result"])


if __name__ == "__main__":
    main()
