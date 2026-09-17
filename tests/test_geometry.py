import pytest
import torch
import torch.nn.functional as F
from tasktext.geometry import text_geometry, affine_statistics, coordinate_logits, nearest_rank
from tasktext.adapters import EndpointAdapter, ENDPOINT_METHODS
from tasktext.training import predict, candidate_grid, selection_key, location_for
from tasktext.baselines import kernel, select_proker


def tensors():
    g = torch.Generator().manual_seed(8021)
    rand = lambda *s: torch.randn(*s, generator=g)
    return F.normalize(rand(4, 24), dim=-1), rand(7, 32), rand(7, 24), rand(32, 24)


@pytest.mark.parametrize("random", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_compact_probabilities_loss_and_gradient(random, fixed):
    text, h, z0, _ = tensors()
    geometry = text_geometry(text, random=random)
    frozen = affine_statistics(z0, geometry)
    a = torch.randn(32, 3, generator=torch.Generator().manual_seed(1)).mul(.02).requires_grad_()
    compact = coordinate_logits(a, h, frozen, 7., fixed)
    z = z0 + (h @ a) @ geometry["Q"].float().T
    explicit = 7. * ((z / (z0 if fixed else z).norm(dim=-1, keepdim=True)) @ text.T)
    y = torch.arange(7) % 4
    l1, l2 = F.cross_entropy(compact, y), F.cross_entropy(explicit, y)
    g1, = torch.autograd.grad(l1, a, retain_graph=True)
    g2, = torch.autograd.grad(l2, a)
    torch.testing.assert_close(compact.softmax(-1), explicit.softmax(-1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(l1, l2, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(g1, g2, atol=2e-6, rtol=2e-5)


def test_fixed_denominator_r2_p0_fp64():
    text, h, z0, _ = tensors()
    h, z0 = h.double(), z0.double()
    aligned, random = text_geometry(text), text_geometry(text, random=True)
    q, u, td = aligned["Q"], random["Q"], aligned["centered"]
    k = u.T @ q
    g = torch.randn(32, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)).requires_grad_()
    a = torch.linalg.solve(k.T, g.T).T
    denominator = z0.norm(dim=-1, keepdim=True)
    l0 = ((z0 + h @ g @ q.T) @ td) / denominator
    lr = ((z0 + h @ a @ u.T) @ td) / denominator
    torch.testing.assert_close(l0, lr, atol=1e-12, rtol=1e-12)
    y = torch.arange(7) % 4
    loss0, lossr = F.cross_entropy(l0, y) + g.square().sum(), F.cross_entropy(lr, y) + g.square().sum()
    grad0, = torch.autograd.grad(loss0, g, retain_graph=True)
    gradr, = torch.autograd.grad(lossr, g)
    torch.testing.assert_close(grad0, gradr, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("method", ENDPOINT_METHODS)
def test_zero_update_and_export(method):
    text, h, z0, w0 = tensors()
    model = EndpointAdapter(method, text, w0, 7.)
    expected = 7. * F.normalize(z0, dim=-1) @ text.T
    torch.testing.assert_close(model(h, z0), expected)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.001)
    packet = model.export("projector")
    actual = predict(packet, {"h": h, "z0": z0})
    torch.testing.assert_close(actual, model(h, z0), atol=2e-5, rtol=2e-5)


def test_complementary_update_and_displacement():
    text, _, _, w0 = tensors()
    model = EndpointAdapter("comp_rank", text, w0, 7.)
    with torch.no_grad():
        model.B.fill_(.03)
    leading_u, _, leading_vh = torch.linalg.svd(w0, full_matrices=True)
    delta = model.increment()
    assert (leading_u[:, :16].T @ delta).abs().max() < 1e-6
    assert (delta @ leading_vh[:16].T).abs().max() < 1e-6
    torch.testing.assert_close(model.penalty(), delta.square().sum(), rtol=2e-5, atol=1e-7)


def test_selection_budget_and_location():
    assert len(candidate_grid(4)) == len(candidate_grid(16)) == 9
    assert selection_key(5, .001, .25, 4) > selection_key(5, .0001, .25, 4)
    assert selection_key(5, .001, 100., 4) > selection_key(5, .001, 1., 4)
    assert nearest_rank(6912, 1536) == 4
    assert nearest_rank(6912, 3808) == 2
    assert location_for("p0", "siglip2") == "output"
    assert location_for("comp_param", "siglip2") == "fc2"
    with pytest.raises(ValueError):
        location_for("comp_param", "siglip2", "output")


def test_proker_matches_dense_solve():
    text, _, z0, _ = tensors()
    x = F.normalize(z0, dim=-1)
    y = torch.arange(len(x)) % len(text)
    packet = select_proker(x, y, x, y, text, "dtd", ([.7], [.3]))
    k = kernel(x.double(), x.double(), .7)
    rhs = F.one_hot(y, len(text)).double() - x.double() @ text.double().T
    reference = torch.linalg.solve(k / .3 + torch.eye(len(x), dtype=torch.float64), rhs)
    torch.testing.assert_close(packet["alpha"], reference, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("dataset,count", [("dtd", 2000), ("eurosat", 2000),
                                          ("oxford_pets", 800), ("sun397", 200)])
def test_full_proker_search_grids(dataset, count):
    text, _, z0, _ = tensors()
    x = F.normalize(z0, dim=-1)
    y = torch.arange(len(x)) % len(text)
    packet = select_proker(x, y, x, y, text, dataset)
    assert len(packet["candidates"]) == count
    assert torch.isfinite(packet["alpha"]).all()
