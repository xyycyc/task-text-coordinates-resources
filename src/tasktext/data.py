"""Image-identity manifests and self-contained tensor caches."""
import hashlib
import json
from pathlib import Path
import zlib
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_manifest(manifest):
    names = manifest["classnames"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Class names must be nonempty and unique")
    sets = {}
    for split in ("train", "val", "test"):
        rows = manifest[split]
        ids = [r[0] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate identities in {split}")
        if any(not 0 <= r[1] < len(names) or Path(r[0]).is_absolute() or ".." in Path(r[0]).parts for r in rows):
            raise ValueError("Invalid relative image identity or class index")
        sets[split] = set(ids)
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Support, validation and test image identities overlap")
    for key, episode in manifest["episodes"].items():
        for source, field in (("train", "train_ids"), ("val", "val_ids")):
            ids = episode[field]
            if len(ids) != len(set(ids)) or not set(ids) <= sets[source]:
                raise ValueError(f"Invalid {key} {field}")
    return manifest


def episode_rows(manifest, shots, seed, split):
    lookup = {r[0]: r for r in manifest[split]}
    if split == "test":
        return manifest["test"]
    episode = manifest["episodes"][f"k{shots}_s{seed}"]
    return [lookup[i] for i in episode["train_ids" if split == "train" else "val_ids"]]


def encode_cache(manifest_path, image_root, output, family, shots, seed, *,
                 device="cpu", checkpoint=None, download_root=None, batch_size=32,
                 splits=("train", "val"), views=32, encoder_factory=None):
    from PIL import Image
    from .encoders import Encoder
    manifest = validate_manifest(read_json(manifest_path))
    if shots not in (4, 16) or seed not in (1, 2, 3):
        raise ValueError("The paper protocol uses shots 4/16 and seeds 1/2/3")
    if not set(splits) <= {"train", "val", "test"}:
        raise ValueError("Unknown cache split")
    output, image_root = Path(output), Path(image_root).resolve()
    if batch_size < 1 or views < 1:
        raise ValueError("batch_size and views must be positive")
    signature = {"manifest_sha256": digest(manifest), "family": family,
                 "dataset": manifest["dataset"], "shots": shots, "seed": seed, "views": views,
                 "encoder_source": str(checkpoint) if checkpoint else family}
    if (output / "base.pt").exists():
        old = load_tensor(output / "base.pt")
        if old["identity"] != signature:
            raise ValueError("Output cache belongs to another episode or encoder")
        if all((output / (s + ".pt")).exists() for s in splits):
            return
    encoder = (encoder_factory or Encoder)(family, manifest["classnames"], manifest["template"],
                                          device, checkpoint, download_root)
    base = dict(signature, identity=signature, text=encoder.text, w0=encoder.w0,
                scale=encoder.scale, classnames=manifest["classnames"],
                prompts=encoder.prompts, checkpoint=encoder.checkpoint_name)
    if (output / "base.pt").exists():
        old = load_tensor(output / "base.pt")
        if not torch.equal(old["text"], base["text"]) or not torch.equal(old["w0"], base["w0"]):
            encoder.close()
            raise ValueError("Encoder tensors differ from the existing cache")
    else:
        save_tensor(output / "base.pt", base)
    try:
        for split in splits:
            path = output / (split + ".pt")
            if path.exists():
                continue
            rows = episode_rows(manifest, shots, seed, split)
            h_parts, z_parts = [], []
            count = views if split == "train" else 1
            # Work in bounded image groups; augmented views never cross image identities.
            for offset in range(0, len(rows), max(1, batch_size // count)):
                group = rows[offset:offset + max(1, batch_size // count)]
                pixels = []
                for name, _ in group:
                    path_image = (image_root / name).resolve()
                    if not path_image.is_relative_to(image_root):
                        raise ValueError("Image path escapes the dataset root")
                    with Image.open(path_image) as source:
                        image = source.convert("RGB")
                        for view in range(count):
                            if split == "train":
                                with torch.random.fork_rng(devices=[]):
                                    torch.random.default_generator.manual_seed(
                                        (17290 + view * 1000003 + zlib.crc32(name.encode())) % 2**32)
                                    pixels.append(encoder.train_transform(image))
                            else:
                                pixels.append(encoder.eval_transform(image))
                hh, zz = [], []
                for first in range(0, len(pixels), batch_size):
                    h, z = encoder.encode(torch.stack(pixels[first:first + batch_size]))
                    hh.append(h); zz.append(z)
                h, z = torch.cat(hh), torch.cat(zz)
                if split == "train":
                    h, z = h.reshape(len(group), count, -1), z.reshape(len(group), count, -1)
                h_parts.append(h); z_parts.append(z)
            packet = {"identity": signature, "h": torch.cat(h_parts), "z0": torch.cat(z_parts),
                      "labels": torch.tensor([r[1] for r in rows], dtype=torch.long),
                      "ids": [r[0] for r in rows], "split": split}
            save_tensor(output / (split + ".pt"), packet)
    finally:
        encoder.close()


def load_part(folder, split, base):
    part = load_tensor(Path(folder) / (split + ".pt"))
    if part["identity"] != base["identity"] or part["split"] != split:
        raise ValueError("Cache identity/split mismatch")
    if len(part["ids"]) != len(set(part["ids"])) or len(part["ids"]) != len(part["labels"]):
        raise ValueError("Cache contains duplicate or missing image identities")
    if part["h"].shape[:-1] != part["z0"].shape[:-1] or len(part["z0"]) != len(part["labels"]):
        raise ValueError("Cache tensor shapes disagree")
    if not torch.isfinite(part["h"]).all() or not torch.isfinite(part["z0"]).all():
        raise ValueError("Cache contains nonfinite features")
    return part
