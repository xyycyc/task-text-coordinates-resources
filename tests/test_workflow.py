import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from tasktext.data import write_json, read_json, validate_manifest, encode_cache, load_tensor, load_part
from tasktext.cli import train_cache, evaluate_cache


class TinyEncoder:
    """Shape-compatible image encoder for testing image/identity plumbing."""
    def __init__(self, family, names, template, device, checkpoint, download_root):
        from torchvision.transforms import Compose, Resize, ToTensor
        generator = torch.Generator().manual_seed(9)
        self.w0 = torch.randn(48, 24, generator=generator) / 48**.5
        self.text = F.normalize(torch.randn(len(names), 24, generator=generator), dim=-1)
        self.prompts = [template.format(n) for n in names]
        self.scale, self.checkpoint_name = 5., "synthetic-test"
        self.train_transform = self.eval_transform = Compose([Resize((4, 4)), ToTensor()])

    def encode(self, images):
        h = images.flatten(1)
        return h, h @ self.w0

    def close(self):
        pass


def manifest(tmp_path):
    obj = {"dataset": "dtd", "classnames": ["one", "two"], "template": "{} texture.",
           "train": [], "val": [], "test": [], "episodes": {}}
    for split, n in [("train", 4), ("val", 4), ("test", 2)]:
        for c in range(2):
            for i in range(n):
                name = f"{split}_{c}_{i}.png"
                Image.new("RGB", (8, 8), (20 + 80*c, 10 + i*15, 40)).save(tmp_path / name)
                obj[split].append([name, c])
    obj["episodes"]["k4_s1"] = {"train_ids": [r[0] for r in obj["train"]],
                               "val_ids": [r[0] for r in obj["val"]]}
    write_json(tmp_path / "manifest.json", obj)
    return obj


def test_images_selection_serialization_and_evaluation(tmp_path):
    obj = manifest(tmp_path)
    cache = tmp_path / "cache"
    encode_cache(tmp_path / "manifest.json", tmp_path, cache, "clip", 4, 1,
                 encoder_factory=TinyEncoder, batch_size=16)
    assert not (cache / "test.pt").exists()
    train_cache(cache, tmp_path / "models", ["p0"], steps=3)
    saved = load_tensor(tmp_path / "models/p0.pt")
    assert saved["selection_complete"]
    assert len(saved["candidates"]) == 9
    encode_cache(tmp_path / "manifest.json", tmp_path, cache, "clip", 4, 1,
                 encoder_factory=TinyEncoder, splits=("test",))
    evaluate_cache(cache, tmp_path / "models/p0.pt", tmp_path / "result.json")
    result = read_json(tmp_path / "result.json")
    assert result["total"] == 4
    assert [r["id"] for r in result["predictions"]] == [r[0] for r in obj["test"]]
    assert 0 <= result["accuracy"] <= 100


def test_manifest_rejects_split_overlap(tmp_path):
    obj = manifest(tmp_path)
    obj["test"][0] = obj["train"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(obj)


def test_committed_splits_and_results():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "data/splits").glob("*.json"))
    assert len(files) == 4
    for path in files:
        obj = validate_manifest(read_json(path))
        assert set(obj["episodes"]) == {f"k{k}_s{s}" for k in (4, 16) for s in (1, 2, 3)}
        for key, ep in obj["episodes"].items():
            k = int(key.split("_")[0][1:])
            assert len(ep["train_ids"]) == k * len(obj["classnames"])
            assert len(ep["val_ids"]) == (4 if obj["dataset"] == "sun397" else k) * len(obj["classnames"])
