"""Validation-only selection and portable prediction packets."""
import torch
import torch.nn.functional as F
from .adapters import EndpointAdapter, COORDINATE_METHODS
from .geometry import affine_statistics, text_geometry
from .baselines import support_features, select_lp, select_lpplusplus, select_proker


def candidate_grid(shots):
    return [(1e-3, lam) for lam in (1 / shots, 0., .01, .1, 1., 10., 100.)] + [
        (1e-4, 1 / shots), (1e-2, 1 / shots)]


def selection_key(correct, lr, penalty, shots):
    return correct, (lr == 1e-3 and penalty == 1 / shots), -lr, penalty


def location_for(method, family, location=None):
    if family == "clip":
        if location not in (None, "projector"):
            raise ValueError("CLIP uses the visual projector")
        return "projector"
    if location is not None:
        if location not in ("output", "fc2"):
            raise ValueError("SigLIP 2 location must be output or fc2")
        if (method == "svd" or method.startswith("comp")) and location != "fc2":
            raise ValueError("The spectral configurations use FC2")
        return location
    return "fc2" if method == "svd" or method.startswith("comp") else "output"


def inputs(part, location):
    return part["z0"] if location == "output" else part["h"]


@torch.no_grad()
def predict(packet, part, device="cpu", batch_size=256):
    """No labels are read here. The packet is fixed before evaluation."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    result = []
    move = lambda t: t.to(device)
    text = move(packet["text"]) if "text" in packet else None
    for start in range(0, len(part["z0"]), batch_size):
        z0 = move(part["z0"][start:start + batch_size])
        if packet["kind"] == "endpoint":
            h = move(inputs(part, packet["location"])[start:start + batch_size])
            z = z0 + h @ move(packet["delta"])
            norm = (z0 if packet["fixed_denominator"] else z).norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = packet["scale"] * ((z / norm) @ text.T)
        elif packet["kind"] == "linear":
            w, b = move(packet["weight"]), move(packet["bias"])
            scores = F.linear(F.normalize(z0, dim=-1).to(w.dtype), w, b)
        elif packet["kind"] == "kernel":
            from .baselines import kernel
            x = F.normalize(z0, dim=-1).double()
            scores = x @ text.T + kernel(x, move(packet["support"]),
                                        packet["config"]["beta"]) @ move(packet["alpha"])
        else:
            raise ValueError("Unknown prediction packet")
        result.append(scores.cpu())
    return torch.cat(result)


def fit_and_select(base, train, val, method, *, device="cpu", steps=300, location=None,
                   fixed_config=False, proker_grid=None):
    """The function accepts support and validation only."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if fixed_config and method not in COORDINATE_METHODS:
        raise ValueError("Fixed-configuration diagnostics apply to coordinate methods")
    text = base["text"].to(device)
    train = {k: v.to(device) if torch.is_tensor(v) else v for k, v in train.items()}
    val = {k: v.to(device) if torch.is_tensor(v) else v for k, v in val.items()}
    if train["labels"].dtype != torch.long or val["labels"].dtype != torch.long:
        raise ValueError("Class labels must be int64")
    if train["labels"].min() < 0 or train["labels"].max() >= len(text):
        raise ValueError("Support labels are outside the class order")
    expected = train["labels"].new_full((len(text),), base["shots"])
    if not torch.equal(torch.bincount(train["labels"], minlength=len(text)), expected):
        raise ValueError("Support must contain K original images per class")
    if val["labels"].min() < 0 or val["labels"].max() >= len(text):
        raise ValueError("Validation labels are outside the class order")
    if method in ("lp", "lpplusplus", "proker"):
        x, y = support_features(train["z0"], train["labels"], method)
        vx, vy = F.normalize(val["z0"], dim=-1), val["labels"]
        if method == "lp":
            return select_lp(x, y, vx, vy)
        if method == "lpplusplus":
            return select_lpplusplus(x, y, vx, vy, text, base["shots"], base["seed"], steps)
        return select_proker(x, y, vx, vy, text, base["dataset"], proker_grid)
    location = location_for(method, base["family"], location)
    w0 = (torch.eye(text.shape[1], device=device) if location == "output" else base["w0"].to(device))
    h = inputs(train, location)
    rank = text_geometry(text)["rank"]
    p0_input = text.shape[1] if base["family"] == "siglip2" else base["w0"].shape[0]
    spectral = (torch.linalg.svd(w0, full_matrices=method.startswith("comp"))
                if method == "svd" or method.startswith("comp") else None)
    grid = [(1e-3, 1 / base["shots"])] if fixed_config else candidate_grid(base["shots"])
    best, best_key, rows = None, None, []
    for lr, penalty in grid:
        model = EndpointAdapter(method, text, w0, base["scale"], base["seed"],
                                p0_input * rank, spectral)
        frozen = affine_statistics(train["z0"], model.geometry) if method in COORDINATE_METHODS else None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        for step in range(steps):
            view = (step + 1) % train["z0"].shape[1]
            scores = model(h[:, view], train["z0"][:, view], frozen, view)
            loss = F.cross_entropy(scores, train["labels"]) + penalty * model.penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite {method} loss/gradient at step {step + 1}")
            optimizer.step()
            scheduler.step()
        packet = model.export(location)
        scores = predict(packet, val, device)
        correct = int((scores.argmax(-1) == val["labels"].cpu()).sum())
        config = {"lr": lr, "lambda": penalty, "steps": steps, "seed": base["seed"]}
        rows.append({**config, "val_correct": correct})
        key = selection_key(correct, lr, penalty, base["shots"])
        if best is None or key > best_key:
            best, best_key = packet, key
            best.update(config=config, val_correct=correct)
    best["candidates"] = rows
    return best
