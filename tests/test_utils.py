"""Tests for the flatten/unflatten parameter utilities.

The three functions share a single contract: flatten_params, flatten_grads and
unflatten_into must all agree on one ordering and one index layout, so that
element i means the same parameter in every vector. Stages 2 and 3 depend on
that: averaging deltas across clients, and applying top-k indices selected on
one model to another, are both meaningless if the layout drifts.
"""

import copy

import pytest
import torch
from torch import nn

from fedrl_bench.utils import flatten_grads, flatten_params, unflatten_into

# The Stage 1 CartPole policy net: 4 -> 64 -> 2.
EXPECTED_NUMEL = 4 * 64 + 64 + 64 * 2 + 2  # 450


def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))


@pytest.fixture
def model() -> nn.Module:
    return make_model()


@pytest.fixture
def trained_model() -> nn.Module:
    """A model that has had backward() called, so every .grad is populated."""
    m = make_model()
    m(torch.randn(8, 4)).sum().backward()
    return m


# ---------------------------------------------------------------------------
# Shape and length
# ---------------------------------------------------------------------------

def test_flatten_params_length(model):
    assert flatten_params(model).numel() == EXPECTED_NUMEL


def test_flatten_params_is_1d(model):
    assert flatten_params(model).dim() == 1


def test_flatten_grads_length(trained_model):
    assert flatten_grads(trained_model).numel() == EXPECTED_NUMEL


def test_flatten_params_matches_sum_of_numels(model):
    expected = sum(p.numel() for p in model.parameters())
    assert flatten_params(model).numel() == expected


# ---------------------------------------------------------------------------
# The length contract: params and grads must always agree
# ---------------------------------------------------------------------------

def test_grads_length_matches_params_before_backward(model):
    """No backward() has run, so every .grad is None. The vector must still be
    full length, or index i means different things in the two vectors."""
    assert flatten_grads(model).numel() == flatten_params(model).numel()


def test_grads_are_zero_before_backward(model):
    assert torch.equal(flatten_grads(model), torch.zeros(EXPECTED_NUMEL))


def test_grads_length_with_frozen_layer():
    """A frozen parameter never receives a gradient. The layout must not shift."""
    m = make_model()
    for p in m[2].parameters():
        p.requires_grad_(False)
    m(torch.randn(8, 4)).sum().backward()
    assert flatten_grads(m).numel() == flatten_params(m).numel()


def test_frozen_layer_grads_are_zero_in_place():
    """The zeros must land in the frozen layer's slice, not be appended."""
    m = make_model()
    for p in m[2].parameters():
        p.requires_grad_(False)
    m(torch.randn(8, 4)).sum().backward()

    frozen_numel = sum(p.numel() for p in m[2].parameters())
    grads = flatten_grads(m)
    assert torch.equal(grads[-frozen_numel:], torch.zeros(frozen_numel))
    assert not torch.equal(
        grads[:-frozen_numel], torch.zeros(EXPECTED_NUMEL - frozen_numel)
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_round_trip_is_exact(model):
    """flatten -> unflatten -> flatten must be bitwise identical, not merely close."""
    original = flatten_params(model).clone()
    unflatten_into(model, original)
    assert torch.equal(flatten_params(model), original)


def test_unflatten_writes_expected_values(model):
    target = torch.arange(EXPECTED_NUMEL, dtype=torch.float32)
    unflatten_into(model, target)
    assert torch.equal(flatten_params(model), target)


def test_unflatten_preserves_per_parameter_shapes(model):
    shapes_before = [tuple(p.shape) for p in model.parameters()]
    unflatten_into(model, torch.randn(EXPECTED_NUMEL))
    assert [tuple(p.shape) for p in model.parameters()] == shapes_before


def test_ordering_is_layerwise(model):
    """Element 0 must be the first weight of the first layer, and the final
    element the last layer's bias: cat order follows parameters() order."""
    target = torch.arange(EXPECTED_NUMEL, dtype=torch.float32)
    unflatten_into(model, target)
    params = list(model.parameters())
    assert params[0].flatten()[0].item() == 0.0
    assert params[-1].flatten()[-1].item() == float(EXPECTED_NUMEL - 1)


# ---------------------------------------------------------------------------
# Autograd state
# ---------------------------------------------------------------------------

def test_flatten_params_is_detached(model):
    """Federated arithmetic (w_local - w_global) must not build a graph."""
    flat = flatten_params(model)
    assert not flat.requires_grad
    assert flat.grad_fn is None


def test_flatten_grads_is_detached(trained_model):
    assert not flatten_grads(trained_model).requires_grad


def test_unflatten_preserves_requires_grad(model):
    unflatten_into(model, torch.randn(EXPECTED_NUMEL))
    assert all(p.requires_grad for p in model.parameters())


def test_unflatten_leaves_model_trainable(model):
    """After loading weights the model must still backprop, i.e. the parameters
    were written to in place rather than replaced by non-leaf tensors."""
    unflatten_into(model, torch.randn(EXPECTED_NUMEL))
    model(torch.randn(8, 4)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_unflatten_does_not_clear_gradients(trained_model):
    """Loading parameters and zeroing gradients are separate operations."""
    before = flatten_grads(trained_model).clone()
    unflatten_into(trained_model, torch.randn(EXPECTED_NUMEL))
    assert torch.equal(flatten_grads(trained_model), before)


# ---------------------------------------------------------------------------
# Aliasing: the server broadcasts one vector to every client
# ---------------------------------------------------------------------------

def test_flatten_params_does_not_alias_model(model):
    flat = flatten_params(model)
    before = flat.clone()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert torch.equal(flat, before)


def test_unflatten_does_not_alias_source_vector(model):
    """If the model aliased the broadcast vector, one client training would
    silently corrupt the global weights and every other client's copy."""
    vec = torch.randn(EXPECTED_NUMEL)
    unflatten_into(model, vec)
    snapshot = flatten_params(model).clone()
    vec.add_(999.0)
    assert torch.equal(flatten_params(model), snapshot)


def test_unflatten_mutates_parameters_in_place(model):
    """Parameter object identity must survive, or an optimiser constructed
    before the call would be left holding stale references."""
    ids_before = [id(p) for p in model.parameters()]
    unflatten_into(model, torch.randn(EXPECTED_NUMEL))
    assert [id(p) for p in model.parameters()] == ids_before


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [-1, 1, -449, 450])
def test_unflatten_rejects_wrong_length(model, delta):
    with pytest.raises(ValueError):
        unflatten_into(model, torch.zeros(EXPECTED_NUMEL + delta))


def test_unflatten_error_message_names_both_sizes(model):
    """The message is the whole point of the check: it must say what was
    expected and what arrived, or it saves no debugging time."""
    with pytest.raises(ValueError) as excinfo:
        unflatten_into(model, torch.zeros(EXPECTED_NUMEL + 7))
    message = str(excinfo.value)
    assert str(EXPECTED_NUMEL) in message
    assert str(EXPECTED_NUMEL + 7) in message


def test_unflatten_rejects_wrong_length_before_mutating(model):
    """A rejected call must leave the model untouched, not half-written."""
    before = flatten_params(model).clone()
    with pytest.raises(ValueError):
        unflatten_into(model, torch.zeros(EXPECTED_NUMEL - 1))
    assert torch.equal(flatten_params(model), before)


# ---------------------------------------------------------------------------
# Cross-model layout: the assumption Stage 2 rests on
# ---------------------------------------------------------------------------

def test_layout_is_stable_across_instances():
    """Two independently constructed models of the same architecture must share
    a layout, or averaging client A's vector with client B's is nonsense."""
    a, b = make_model(seed=0), make_model(seed=1)
    assert flatten_params(a).numel() == flatten_params(b).numel()
    assert [tuple(p.shape) for p in a.parameters()] == [
        tuple(p.shape) for p in b.parameters()
    ]


def test_weights_transfer_between_instances():
    """The Stage 2 broadcast: server vector -> a fresh client model."""
    server, client = make_model(seed=0), make_model(seed=1)
    assert not torch.equal(flatten_params(server), flatten_params(client))

    unflatten_into(client, flatten_params(server))
    assert torch.equal(flatten_params(client), flatten_params(server))


def test_averaging_two_clients_round_trips():
    """A minimal FedAvg step end to end, in the flat representation."""
    a, b, target = make_model(seed=0), make_model(seed=1), make_model(seed=2)
    mean = (flatten_params(a) + flatten_params(b)) / 2
    unflatten_into(target, mean)
    assert torch.allclose(flatten_params(target), mean)


def test_delta_application_matches_direct_training():
    """delta = w_after - w_before, applied to a copy, reproduces the trained model."""
    m = make_model()
    w_before = flatten_params(m).clone()

    m(torch.randn(8, 4)).sum().backward()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(p.grad, alpha=-0.1)

    delta = flatten_params(m) - w_before

    rebuilt = make_model()
    unflatten_into(rebuilt, w_before + delta)
    # allclose, not equal: (w + (w_after - w)) is not bitwise w_after in float32.
    assert torch.allclose(flatten_params(rebuilt), flatten_params(m))


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------

def test_flatten_params_preserves_dtype(model):
    assert flatten_params(model).dtype == next(model.parameters()).dtype


def test_flatten_grads_preserves_dtype(trained_model):
    assert flatten_grads(trained_model).dtype == next(trained_model.parameters()).dtype


def test_round_trip_in_float64():
    m = make_model().double()
    original = flatten_params(m).clone()
    assert original.dtype == torch.float64
    unflatten_into(m, original)
    assert torch.equal(flatten_params(m), original)


# ---------------------------------------------------------------------------
# Larger model, to catch anything that only works at small sizes
# ---------------------------------------------------------------------------

def test_round_trip_on_mnist_sized_model():
    m = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    expected = 784 * 128 + 128 + 128 * 10 + 10
    original = flatten_params(m).clone()
    assert original.numel() == expected

    unflatten_into(m, original)
    assert torch.equal(flatten_params(m), original)


def test_transfer_matches_deepcopy(model):
    """A deepcopy and a flat-vector transfer must produce identical models."""
    clone_a = copy.deepcopy(model)
    clone_b = make_model(seed=99)
    unflatten_into(clone_b, flatten_params(model))
    assert torch.equal(flatten_params(clone_a), flatten_params(clone_b))
