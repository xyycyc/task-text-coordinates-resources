"""P0, full, factorized, spectral and random-coordinate endpoint updates."""
import math
import torch
from torch import nn
from .geometry import text_geometry, nearest_rank, coordinate_logits

ENDPOINT_METHODS = (
    "p0", "prolip", "lora_rank", "lora_param", "svd",
    "comp_k2", "comp_rank", "comp_param", "r0", "r1", "r2",
    "dynamic_ci", "fixed_ci", "fixed_disp",
)
COORDINATE_METHODS = ("p0", "r0", "r1", "r2", "dynamic_ci", "fixed_ci", "fixed_disp")


class EndpointAdapter(nn.Module):
    def __init__(self, method, text, w0, scale, seed=1, p0_budget=None, spectral=None):
        super().__init__()
        if method not in ENDPOINT_METHODS:
            raise ValueError(f"Unknown endpoint method: {method}")
        self.method, self.scale = method, float(scale)
        self.geometry = text_geometry(text, random=method in ("r0", "r1", "r2"))
        self.register_buffer("text", text.detach().float())
        self.register_buffer("w0", w0.detach().float())
        q = self.geometry["Q"]
        self.register_buffer("Q", q)
        self.register_buffer("R", self.geometry["R"].float())
        d, e = w0.shape
        r = q.shape[1]
        budget = d * r if p0_budget is None else p0_budget
        g = torch.Generator(device=text.device).manual_seed(seed)
        zeros = lambda *shape: nn.Parameter(torch.zeros(*shape, device=text.device))
        uniform = lambda rows, cols: nn.Parameter(torch.empty(
            rows, cols, device=text.device).uniform_(-1 / math.sqrt(cols), 1 / math.sqrt(cols), generator=g))
        if method in COORDINATE_METHODS:
            self.A = zeros(d, r)
            if method in ("r1", "r2"):
                self.register_buffer("K", (q.T @ text_geometry(text)["Q"]).float())
        elif method == "prolip":
            self.delta = zeros(d, e)
        elif method.startswith("lora"):
            k = r if method == "lora_rank" else nearest_rank(budget, d + e)
            self.B, self.C = zeros(d, k), uniform(k, e)
        else:
            full = method.startswith("comp")
            u, s, vh = (spectral if spectral is not None else
                        torch.linalg.svd(w0, full_matrices=full))
            if method == "svd":
                self.register_buffer("U", u[:, :len(s)])
                self.register_buffer("Vh", vh[:len(s)])
                self.s = zeros(len(s))
            else:
                if min(d, e) <= 16:
                    raise ValueError("Comp-E requires endpoint dimensions greater than 16")
                k = (2 if method == "comp_k2" else r if method == "comp_rank"
                     else nearest_rank(budget, d + e - 32))
                if k > min(d - 16, e - 16):
                    raise ValueError("Requested Comp-E rank exceeds the complementary space")
                self.register_buffer("U", u[:, 16:])
                self.register_buffer("Vh", vh[16:])
                self.B, self.C = zeros(d - 16, k), uniform(k, e - 16)

    @property
    def fixed_denominator(self):
        return self.method in ("fixed_ci", "fixed_disp")

    def coordinates(self):
        return torch.linalg.solve(self.K.T, self.A.T).T if self.method == "r2" else self.A

    def increment(self, dtype=None):
        cv = lambda x: x if dtype is None else x.to(dtype)
        if self.method in COORDINATE_METHODS:
            return cv(self.coordinates()) @ self.Q.to(dtype or self.A.dtype).T
        if self.method == "prolip":
            return cv(self.delta)
        if self.method.startswith("lora"):
            return cv(self.B) @ cv(self.C)
        if self.method == "svd":
            return (cv(self.U) * cv(self.s)) @ cv(self.Vh)
        return ((cv(self.U) @ cv(self.B)) @ cv(self.C)) @ cv(self.Vh)

    def penalty(self):
        if self.method in ("r1", "r2"):
            return (self.A @ self.K if self.method == "r1" else self.A).square().sum()
        if self.method in ("dynamic_ci", "fixed_ci"):
            return (self.A @ self.R).square().sum() / self.geometry["gamma"].float()
        if self.method in COORDINATE_METHODS:
            return self.A.square().sum()
        if self.method == "prolip":
            return self.delta.square().sum()
        if self.method == "svd":
            return self.s.square().sum()
        return (self.B @ self.C).square().sum()

    def forward(self, h, z0, frozen=None, view=None):
        if self.method in COORDINATE_METHODS and frozen is not None:
            return coordinate_logits(self.coordinates(), h, frozen, self.scale,
                                     self.fixed_denominator, view)
        if self.method.startswith("lora"):
            z = z0 + (h @ self.B) @ self.C
        elif self.method.startswith("comp"):
            z = z0 + (((h @ self.U) @ self.B) @ self.C) @ self.Vh
        elif self.method == "svd":
            z = z0 + ((h @ self.U) * self.s) @ self.Vh
        else:
            z = z0 + h @ self.increment()
        norm = (z0 if self.fixed_denominator else z).norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self.scale * ((z / norm) @ self.text.T)

    @torch.no_grad()
    def export(self, location):
        return {"kind": "endpoint", "method": self.method, "location": location,
                "delta": self.increment(torch.float64).float().cpu(),
                "text": self.text.cpu(), "scale": self.scale,
                "fixed_denominator": self.fixed_denominator,
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "parameters": {name: p.detach().cpu() for name, p in self.named_parameters()}}
