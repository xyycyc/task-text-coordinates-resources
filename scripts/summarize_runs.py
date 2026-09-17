"""Summarize complete, seed-paired results from run_matrix.py."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from tasktext.data import read_json, write_json


def summarize(root):
    plan = read_json(Path(root) / "task_matrix.json")
    rows = defaultdict(dict)
    for job in plan:
        result = read_json(job["result"])
        expected = (job["family"], job["dataset"], job["shots"], job["seed"], job["fitting_method"])
        actual = tuple(result[k] for k in ("family", "dataset", "shots", "seed", "method"))
        if actual != expected:
            raise ValueError("Result identity disagrees with task matrix")
        key = job["family"], job["dataset"], job["shots"], job["group"], job["method"]
        if job["seed"] in rows[key]:
            raise ValueError("Duplicate support seed")
        rows[key][job["seed"]] = result["accuracy"]
    summary = []
    for key, values in sorted(rows.items()):
        if set(values) != {1, 2, 3}:
            raise ValueError(f"Incomplete three-seed result: {key}")
        sequence = [values[s] for s in (1, 2, 3)]
        summary.append(dict(zip(("family", "dataset", "shots", "group", "method"), key),
                            values=sequence, mean=statistics.mean(sequence),
                            sample_sd=statistics.stdev(sequence)))
    paired = []
    main = {key: values for key, values in rows.items() if key[3] == "main"}
    for key, values in sorted(main.items()):
        if key[4] == "p0":
            continue
        p0 = main[(*key[:4], "p0")]
        differences = [p0[s] - values[s] for s in (1, 2, 3)]
        paired.append(dict(zip(("family", "dataset", "shots", "group", "method"), key),
                           values=differences, mean=statistics.mean(differences),
                           sample_sd=statistics.stdev(differences)))
    return {"seeds": [1, 2, 3], "tables": summary, "p0_minus_comparator": paired}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = summarize(args.input)
    write_json(args.output, result)
    print(f"Summarized {len(result['tables'])} three-seed entries")
