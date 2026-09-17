"""Command-line entry points; CPU is the default execution device."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from .adapters import ENDPOINT_METHODS
from .data import read_json, write_json, digest, save_tensor, load_tensor, load_part, encode_cache
from .training import fit_and_select, predict

METHODS = ENDPOINT_METHODS + ("lp", "lpplusplus", "proker")


def configure(device, threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Supported devices are cpu and cuda[:index]")
    torch.set_num_threads(threads)
    if device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def train_cache(cache, output, methods, device="cpu", steps=300, location=None, fixed=False):
    cache, output = Path(cache), Path(output)
    base = load_tensor(cache / "base.pt")
    train, val = load_part(cache, "train", base), load_part(cache, "val", base)
    if set(train["ids"]) & set(val["ids"]):
        raise ValueError("Support and validation identities overlap")
    if train["z0"].shape[1] != 32:
        raise ValueError("Paper training requires all 32 support views")
    for method in methods:
        name = method + ("_" + location if location else "") + ("_fixed_config" if fixed else "")
        target = output / (name + ".pt")
        if target.exists():
            raise FileExistsError(f"Choose a new output directory: {target}")
        packet = fit_and_select(base, train, val, method, device=device, steps=steps,
                               location=location, fixed_config=fixed)
        packet.update(cache_identity=base["identity"], selection_complete=True,
                      support_ids=train["ids"], validation_ids=val["ids"])
        save_tensor(target, packet)
        write_json(target.with_suffix(".json"), {
            k: packet[k] for k in ("method", "config", "val_correct", "trainable_parameters", "candidates")
        })
        print(f"{name}: validation {packet['val_correct']}/{len(val['labels'])}; saved {target}", flush=True)


def evaluate_cache(cache, model, output, device="cpu", batch_size=256):
    cache = Path(cache)
    packet, base = load_tensor(model), load_tensor(cache / "base.pt")
    if packet.get("selection_complete") is not True or packet["cache_identity"] != base["identity"]:
        raise ValueError("A selected model from the same episode is required")
    test = load_part(cache, "test", base)
    if set(test["ids"]) & (set(packet["support_ids"]) | set(packet["validation_ids"])):
        raise ValueError("Test identities overlap support or validation")
    scores = predict(packet, test, device, batch_size)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite test scores")
    labels = scores.argmax(-1)
    correct = int((labels == test["labels"]).sum())
    result = {"family": base["family"], "dataset": base["dataset"], "shots": base["shots"],
              "seed": base["seed"], "method": packet["method"], "config": packet["config"],
              "correct": correct, "total": len(labels), "accuracy": 100 * correct / len(labels),
              "trainable_parameters": packet["trainable_parameters"],
              "predictions": [{"id": name, "target": int(target), "prediction": int(pred)}
                              for name, target, pred in zip(test["ids"], test["labels"], labels)]}
    write_json(output, result)
    print(f"{packet['method']}: test {result['accuracy']:.4f}% ({correct}/{len(labels)})", flush=True)


def smoke(output, steps=12):
    """A complete synthetic CPU run, including every reported parameterization."""
    configure("cpu", 2)
    output = Path(output)
    generator = torch.Generator().manual_seed(1709)
    classes, e, d, shots, views = 4, 24, 32, 4, 32
    rand = lambda *shape: torch.randn(*shape, generator=generator)
    text = F.normalize(rand(classes, e), dim=-1)
    w0 = rand(d, e) / d**.5
    base = {"text": text, "w0": w0, "scale": 8., "family": "clip",
            "dataset": "dtd", "shots": shots, "seed": 1}
    def part(count, augment=False):
        y = torch.arange(count) % classes
        h = rand(count, views, d) if augment else rand(count, d)
        z = h @ w0 + (text[y, None] if augment else text[y])
        return {"h": h, "z0": z, "labels": y}
    train, val, test = part(classes * shots, True), part(8), part(12)
    records = []
    for method in METHODS:
        packet = fit_and_select(base, train, val, method, steps=steps,
                               proker_grid=([.1, 1.], [.01, 1.]))
        path = output / (method + ".pt")
        save_tensor(path, packet)
        restored = load_tensor(path)
        scores = predict(restored, test)
        if not torch.isfinite(scores).all() or not torch.equal(scores, predict(packet, test)):
            raise AssertionError(f"{method} serialization or finite-score check failed")
        records.append({"method": method, "parameters": packet["trainable_parameters"],
                        "candidates": len(packet["candidates"]), "finite": True})
        print(f"CPU smoke: {method} passed", flush=True)
    if torch.cuda.is_initialized():
        raise AssertionError("The CPU smoke run initialized CUDA")
    write_json(output / "summary.json", {
        "scope": "synthetic software validation", "device": "cpu",
        "cuda_initialized": False, "steps": steps, "methods": records})


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tasktext")
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode", help="Encode one image-identity episode")
    encode.add_argument("--manifest", required=True)
    encode.add_argument("--data-root", required=True, help="Directory containing the manifest's relative images")
    encode.add_argument("--output", required=True)
    encode.add_argument("--family", choices=("clip", "siglip2"), required=True)
    encode.add_argument("--shots", type=int, choices=(4, 16), required=True)
    encode.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    encode.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    encode.add_argument("--checkpoint", help="Local CLIP .pt or SigLIP 2 snapshot; defaults to public checkpoints")
    encode.add_argument("--download-root", help="Cache directory for public model weights/tokenizers")
    encode.add_argument("--batch-size", type=int, default=32)
    train = sub.add_parser("train", help="Train and select using support/validation only")
    train.add_argument("--cache", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--methods", nargs="+", choices=METHODS, default=["p0"])
    train.add_argument("--location", choices=("projector", "output", "fc2"))
    train.add_argument("--fixed-config", action="store_true", help="Use lr=.001, lambda=1/K for coordinate diagnostics")
    evaluate = sub.add_parser("evaluate", help="Evaluate a selected model on the test split")
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int, default=256)
    for p in (encode, train, evaluate):
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=2)
    demo = sub.add_parser("smoke", help="Exercise all methods on synthetic CPU tensors")
    demo.add_argument("--output", default="outputs/cpu-smoke")
    demo.add_argument("--steps", type=int, default=12)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        smoke(args.output, args.steps)
        return
    configure(args.device, args.threads)
    if args.command == "encode":
        encode_cache(args.manifest, args.data_root, args.output, args.family, args.shots, args.seed,
                     device=args.device, checkpoint=args.checkpoint, download_root=args.download_root,
                     batch_size=args.batch_size, splits=tuple(args.splits))
    elif args.command == "train":
        train_cache(args.cache, args.output, args.methods, args.device,
                    location=args.location, fixed=args.fixed_config)
    else:
        evaluate_cache(args.cache, args.model, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
