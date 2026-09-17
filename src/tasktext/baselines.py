"""Frozen-feature LP, LP++ and ProKeR, on an explicitly selected device."""
import torch
import torch.nn.functional as F

LP_GRID = (1e-6, 1e-4, 1e-2, .316, 1., 1e2, 1e4, 1e6)
PROKER_RANGES = {
    "dtd": ((.1, 10., 20), (.001, 10., 100)),
    "eurosat": ((.5, 3., 20), (.001, 1., 100)),
    "oxford_pets": ((.01, 5., 40), (.01, .3, 20)),
    "sun397": ((.1, 10., 20), (.01, 10., 10)),
}


def support_features(z, labels, method):
    if method == "proker":
        if z.shape[1] < 10:
            raise ValueError("ProKeR requires ten support views")
        return F.normalize(z[:, :10].transpose(0, 1).reshape(-1, z.shape[-1]), dim=-1), labels.repeat(10)
    return F.normalize(z[:, 0], dim=-1), labels


def lp_fit(x, y, vx, vy, c):
    x, vx = x.double(), vx.double()
    classes = int(y.max()) + 1
    w = torch.nn.Parameter(x.new_zeros(classes, x.shape[1]))
    b = torch.nn.Parameter(x.new_zeros(classes))
    opt = torch.optim.LBFGS([w, b], lr=1., max_iter=100, max_eval=500,
                           history_size=10, line_search_fn="strong_wolfe",
                           tolerance_grad=1e-8, tolerance_change=1e-12)
    calls = 0

    class EvaluationLimit(Exception):
        pass

    def closure(final=False):
        nonlocal calls
        if calls >= 5000 - (0 if final else 2):
            raise EvaluationLimit()
        opt.zero_grad(set_to_none=True)
        value = x.new_zeros(())
        for start in range(0, len(y), 2048):
            term = F.cross_entropy(F.linear(x[start:start + 2048], w, b),
                                   y[start:start + 2048], reduction="sum") / len(y)
            term.backward()
            value += term.detach()
        penalty = w.square().sum() / (2 * c * len(y))
        penalty.backward()
        calls += 1
        if not torch.isfinite(value + penalty) or not all(torch.isfinite(p.grad).all() for p in (w, b)):
            raise FloatingPointError("Nonfinite LP objective or gradient")
        return value + penalty.detach()

    for _ in range(1000):
        previous = int(opt.state[w].get("n_iter", 0))
        remaining = 1000 - previous
        if remaining <= 0 or calls >= 4998:
            break
        inner = min(100, remaining)
        opt.param_groups[0].update(max_iter=inner, max_eval=min(5 * inner, 4999 - calls))
        backup = [p.detach().clone() for p in (w, b)]
        try:
            opt.step(closure)
        except EvaluationLimit:
            with torch.no_grad():
                w.copy_(backup[0]); b.copy_(backup[1])
            break
        value = closure(final=True)
        gradient = max(float(p.grad.abs().max()) for p in (w, b))
        if gradient <= 1e-8 or int(opt.state[w]["n_iter"]) - previous < inner:
            break
    value = closure(final=True) if calls < 5000 else value
    gradient = max(float(p.grad.abs().max()) for p in (w, b))
    scaled = gradient * max(1., *(float(p.detach().abs().max()) for p in (w, b))) / max(1., abs(float(value)))
    if scaled > 1e-5:
        raise RuntimeError(f"LP did not converge: scaled gradient {scaled:.3g}")
    with torch.no_grad():
        correct = int((F.linear(vx, w, b).argmax(-1) == vy).sum())
    return {"kind": "linear", "method": "lp", "weight": w.detach().cpu(), "bias": b.detach().cpu(),
            "config": {"C": c, "scaled_gradient": scaled}, "val_correct": correct,
            "trainable_parameters": w.numel() + b.numel()}


def select_lp(x, y, vx, vy, grid=LP_GRID):
    best, candidates = None, []
    for c in grid:
        fitted = lp_fit(x, y, vx, vy, c)
        candidates.append({"C": c, "val_correct": fitted["val_correct"]})
        if best is None or (fitted["val_correct"], -c) > (best["val_correct"], -best["config"]["C"]):
            best = fitted
    best["candidates"] = candidates
    return best


def select_lpplusplus(x, y, vx, vy, text, shots, seed, epochs=300):
    x, vx, text = x.float(), vx.float(), text.float()
    n, e = x.shape
    classes = len(text)
    if not torch.equal(torch.bincount(y, minlength=classes), y.new_full((classes,), shots)):
        raise ValueError("LP++ requires K support images per class")
    # Seed the CPU and requested device without initializing any other device.
    with torch.random.fork_rng(devices=[x.device.index or 0] if x.is_cuda else []):
        torch.random.default_generator.manual_seed(seed)
        if x.is_cuda:
            with torch.cuda.device(x.device):
                torch.cuda.manual_seed(seed)
        classifier = torch.nn.Linear(e, classes, device=x.device)
    onehot = F.one_hot(y, classes).to(x.dtype)
    with torch.no_grad():
        classifier.weight.copy_(onehot.T @ x)
    eigenvalues, _ = torch.linalg.eigh(x.T @ x)
    lr_w = float(4 * n / eigenvalues[-1])
    similarity = x @ text.T
    alignment = (onehot * similarity / onehot.sum(0, keepdim=True)).sum(0)
    alpha = torch.nn.Parameter(((250. / shots) * (alignment.double() * shots)).mean().float().repeat(classes))
    lr_alpha = n / (4 * similarity.square().sum(0).max())
    opt = torch.optim.SGD(classifier.parameters(), lr=lr_w, momentum=.9)
    best, trace = None, []
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(x) + similarity * alpha.detach(), y)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            alpha_loss = F.cross_entropy(classifier(x).detach() + similarity * alpha, y)
            grad, = torch.autograd.grad(alpha_loss, alpha)
            with torch.no_grad():
                alpha.sub_(lr_alpha * grad)
        with torch.no_grad():
            weight = classifier.weight + alpha[:, None] * text
            scores = F.linear(vx, weight, classifier.bias)
            if not torch.isfinite(scores).all() or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite LP++ fit")
            correct = int((scores.argmax(-1) == vy).sum())
            trace.append({"epoch": epoch, "val_correct": correct})
            if best is None or correct >= best["val_correct"]:
                best = {"kind": "linear", "method": "lpplusplus", "weight": weight.cpu().clone(),
                        "bias": classifier.bias.detach().cpu().clone(),
                        "classifier_weight": classifier.weight.detach().cpu().clone(),
                        "alpha": alpha.detach().cpu().clone(), "val_correct": correct,
                        "config": {"selected_epoch": epoch, "epochs": epochs, "lr_w": lr_w,
                                   "lr_alpha": float(lr_alpha), "seed": seed},
                        "trainable_parameters": classes * (e + 2)}
    best["candidates"] = trace
    return best


def kernel(x, support, beta):
    return (-float(beta) * (1 - x @ support.T)).exp()


@torch.no_grad()
def select_proker(x, y, vx, vy, text, dataset, grid=None):
    x, vx, text = x.double(), vx.double(), text.double()
    if grid is None:
        b, l = PROKER_RANGES[dataset]
        grid = (torch.linspace(*b, dtype=torch.float32).tolist(),
                torch.linspace(*l, dtype=torch.float32).tolist())
    rhs = F.one_hot(y, len(text)).double() - x @ text.T
    prior = vx @ text.T
    best, rows = None, []
    for beta in grid[0]:
        k = kernel(x, x, beta)
        eigenvalues, vectors = torch.linalg.eigh(k)
        projected = vectors.T @ rhs
        val_projected = kernel(vx, x, beta) @ vectors
        for lam in grid[1]:
            coefficients = projected * (lam / (eigenvalues + lam)).unsqueeze(1)
            scores = prior + val_projected @ coefficients
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite ProKeR solve")
            correct = int((scores.argmax(-1) == vy).sum())
            rows.append({"beta": beta, "lambda": lam, "val_correct": correct})
            if best is None or correct > best["val_correct"]:
                best = {"kind": "kernel", "method": "proker", "support": x.cpu(),
                        "alpha": (vectors @ coefficients).cpu(), "text": text.cpu(),
                        "config": {"beta": beta, "lambda": lam}, "val_correct": correct,
                        "trainable_parameters": 0, "fitted_coefficients": rhs.numel()}
    best["candidates"] = rows
    return best
