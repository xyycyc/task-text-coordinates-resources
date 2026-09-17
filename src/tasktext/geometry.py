"""Fixed task-text geometry and exact normalized scores."""
import torch


@torch.no_grad()
def text_geometry(text, random=False, seed=8675309):
    """text: unit-normalized [classes, embedding] vectors."""
    text = text.detach().double().T
    if text.ndim != 2 or not torch.isfinite(text).all():
        raise ValueError("Expected finite class-text vectors")
    if not torch.allclose(text.norm(dim=0), torch.ones_like(text[0]), atol=3e-4, rtol=3e-4):
        raise ValueError("Class-text vectors must be unit normalized")
    centered = text - text.mean(dim=1, keepdim=True)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * torch.finfo(torch.float64).eps * s[0]
    rank = int((s > tolerance).sum())
    if not 0 < rank <= min(text.shape[0], text.shape[1] - 1):
        raise ValueError("The centered text matrix must have positive rank")
    q = u[:, :rank]
    q = q * q[q.abs().argmax(dim=0), torch.arange(rank, device=q.device)].sign()
    if random:
        generator = torch.Generator(device=text.device).manual_seed(seed)
        q, triangular = torch.linalg.qr(torch.randn(
            text.shape[0], rank, dtype=torch.float64, device=text.device, generator=generator))
        sign = triangular.diag().sign()
        sign[sign == 0] = 1
        q = q * sign
    r = q.T @ centered
    return {"Q": q, "R": r, "centered": centered, "rank": rank,
            "gamma": r.square().sum() / rank, "random": random}


@torch.no_grad()
def affine_statistics(z0, geometry):
    """Cache frozen parallel, perpendicular and native-norm terms."""
    q = geometry["Q"]
    z = z0.double()
    u = z @ q
    perpendicular = z - u @ q.T
    return {"u0": u.float(), "q0": perpendicular.square().sum(-1, keepdim=True).float(),
            "norm0": z.norm(dim=-1, keepdim=True).float(),
            "b0": (perpendicular @ geometry["centered"]).float()
                  if geometry["random"] else None,
            "R": geometry["R"].float()}


def coordinate_logits(a, h, frozen, scale, fixed=False, view=None):
    select = lambda t: t if view is None or t is None else t[:, view]
    u = select(frozen["u0"]) + h @ a
    numerator = u @ frozen["R"]
    if frozen["b0"] is not None:
        numerator = numerator + select(frozen["b0"])
    denominator = (select(frozen["norm0"]) if fixed else
                   (u.square().sum(-1, keepdim=True) + select(frozen["q0"])).sqrt())
    return float(scale) * numerator / denominator.clamp_min(1e-12)


def nearest_rank(budget, scalars_per_rank):
    lower = max(1, budget // scalars_per_rank)
    return min((lower, lower + 1), key=lambda k: (abs(k * scalars_per_rank - budget), k))
