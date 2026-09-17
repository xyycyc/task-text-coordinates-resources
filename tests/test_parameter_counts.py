import json
from pathlib import Path
import torch
import torch.nn.functional as F
from tasktext.adapters import EndpointAdapter


def test_endpoint_parameter_counts_match_all_published_budgets():
    root = Path(__file__).resolve().parents[1]
    cells = json.loads((root / "data/parameter_budgets.json").read_text())["cells"]
    generator = torch.Generator().manual_seed(529)
    checked = 0
    for family, e, dc in (("clip", 512, 768), ("siglip2", 768, 3072)):
        w = torch.randn(dc, e, generator=generator) / dc**.5
        spectral = torch.linalg.svd(w, full_matrices=True)
        main_w = w if family == "clip" else torch.eye(e)
        texts = {}
        for cell in [r for r in cells if r["family"] == family]:
            method, classes = cell["method"], cell["classes"]
            if method in ("lp", "lpplusplus", "proker"):
                continue
            if classes not in texts:
                texts[classes] = F.normalize(torch.randn(classes, e, generator=generator), dim=-1)
            is_spectral = method == "svd" or method.startswith("comp")
            model = EndpointAdapter(method, texts[classes], w if is_spectral else main_w, 100.,
                                    p0_budget=768 * cell["text_rank"],
                                    spectral=spectral if is_spectral else None)
            assert sum(p.numel() for p in model.parameters()) == cell["parameters"], cell
            checked += 1
    assert checked == 64
