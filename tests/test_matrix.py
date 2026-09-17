import importlib.util
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_matrix_covers_final_protocol_and_identical_configs(tmp_path):
    config = json.loads((ROOT / "configs/paper.json").read_text())
    plan = module("run_matrix").jobs(config, tmp_path)
    assert len({j["cache"] for j in plan}) == 48
    main = [j for j in plan if j["group"] == "main"]
    assert len(main) == 522
    assert len({j["model"] for j in main}) == 516
    assert len([j for j in plan if j["group"] == "position"]) == 24
    assert len([j for j in plan if j["group"] == "selected_coordinates"]) == 36
    assert len([j for j in plan if j["group"] == "fixed_coordinates"]) == 48
    assert len([j for j in plan if j["group"] == "normalization"]) == 36
    assert {j["seed"] for j in plan} == {1, 2, 3}
    aliases = [j for j in main if j["method"] != j["fitting_method"]]
    assert len(aliases) == 6
    assert all(j["family"] == "siglip2" and j["dataset"] == "eurosat" for j in aliases)


def test_aggregation_preserves_alias_and_seed_pairing(tmp_path):
    from tasktext.data import write_json
    config = json.loads((ROOT / "configs/paper.json").read_text())
    config.update(shots=[4], methods=["p0", "comp_k2", "comp_param"])
    plan = [j for j in module("run_matrix").jobs(config, tmp_path, ["siglip2"], ["eurosat"])
            if j["group"] == "main"]
    write_json(tmp_path / "task_matrix.json", plan)
    for job in plan:
        write_json(job["result"], {"family": job["family"], "dataset": job["dataset"],
                   "shots": job["shots"], "seed": job["seed"], "method": job["fitting_method"],
                   "accuracy": 80. + job["seed"] if job["method"] == "p0" else 70. + job["seed"]})
    result = module("summarize_runs").summarize(tmp_path)
    assert len(result["tables"]) == 3
    assert all(r["values"] == [10., 10., 10.] for r in result["p0_minus_comparator"])
