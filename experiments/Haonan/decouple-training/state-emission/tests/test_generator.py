# tests/test_generator.py
"""Unit tests for the candidate generator behind the shared sample() contract.
Tests observe core/generator.py externally; the model-bound AdapterGenerator.sample path is NOT
exercised here (no model load) — it is covered indirectly via the pure multinomial_sample RNG core
and the model-free StubGenerator. The contract under test: sample() returns N candidate TEXTS and
no public method (or its return) exposes a logits tensor."""
import importlib.util as ilu
import inspect
import typing
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def bp(n, p):
    s = ilu.spec_from_file_location(n, str(p)); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


G = bp("generator", HERE.parent / "core/generator.py")


def test_multinomial_sample_deterministic():
    logits = torch.tensor([2.0, 1.0, 0.0])
    a = G.multinomial_sample(logits, 16, 1.0, seed=0)
    b = G.multinomial_sample(logits, 16, 1.0, seed=0)
    assert a == b and len(a) == 16 and set(a) <= {0, 1, 2}


def test_render_state_injection():
    # a FakeGenerator built from a stub render_state + stub sampler returns N texts and no logits
    calls = {}
    def render(state): calls["seen"] = state; return f"PROMPT::{state['id']}"
    gen = G.StubGenerator(render_state=render, scripted=["x", "y", "z"])
    out = gen.sample({"id": "k3_j2_0"}, N=3, temperature=1.0, seed=0)
    assert out == ["x", "y", "z"] and calls["seen"]["id"] == "k3_j2_0"
    assert not hasattr(gen.sample, "logits")  # contract: texts only


def test_sample_returns_exactly_N_strings():
    # StubGenerator must hand back exactly N candidate texts (truncating a longer script).
    gen = G.StubGenerator(render_state=lambda s: "p", scripted=["a", "b", "c", "d", "e"])
    out = gen.sample({"id": "x"}, N=3, temperature=1.0, seed=0)
    assert len(out) == 3
    assert all(isinstance(t, str) for t in out)
    assert out == ["a", "b", "c"]


def test_adaptergenerator_sample_typed_to_return_list():
    # The shared contract advertises a list-of-texts return, never a tensor of logits.
    sig = inspect.signature(G.AdapterGenerator.sample)
    hints = typing.get_type_hints(G.AdapterGenerator.sample)
    ann = hints.get("return", sig.return_annotation)
    assert ann in (list, typing.List, "list") or getattr(ann, "__origin__", None) is list, (
        f"AdapterGenerator.sample return annotation should be list, got {ann!r}")
    # Generator protocol advertises the same list return.
    psig = inspect.signature(G.Generator.sample)
    assert psig.return_annotation in (list, "list") or getattr(
        psig.return_annotation, "__origin__", None) is list


def test_no_public_method_returns_logits_tensor():
    # Contract: no public method name leaks logits, and no public method statically returns a Tensor.
    for cls in (G.AdapterGenerator, G.StubGenerator):
        public = [n for n in dir(cls) if not n.startswith("_")]
        assert not any("logit" in n.lower() for n in public), (
            f"{cls.__name__} exposes a logits-named public method: {public}")
        for name in public:
            attr = getattr(cls, name)
            if not callable(attr):
                continue
            try:
                ann = inspect.signature(attr).return_annotation
            except (ValueError, TypeError):
                continue
            assert ann is not torch.Tensor, (
                f"{cls.__name__}.{name} statically returns a Tensor — logits must never be exposed")
