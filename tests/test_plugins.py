import importlib.metadata as md

import numpy as np
import pytest

from drbench import methods


class _FakeEntryPoint:
    def __init__(self, name, load):
        self.name, self.value, self.load = name, f"fake:{name}", load


def test_plugins_register_methods_and_broken_ones_only_warn(monkeypatch):
    def good():
        @methods.register("plugin_first_coords")
        def first_coords(X, seed=0):
            return np.asarray(X)[:, :2]

    def bad():
        raise ImportError("missing dependency")

    eps = [_FakeEntryPoint("good", good), _FakeEntryPoint("bad", bad)]
    monkeypatch.setattr(md, "entry_points", lambda group: eps if group == "drbench.methods" else [])
    with pytest.warns(UserWarning, match="bad"):
        methods._load_plugins()
    assert "plugin_first_coords" in methods.available()
    X = np.arange(12.0).reshape(4, 3)
    assert np.array_equal(methods.run("plugin_first_coords", X), X[:, :2])
    methods._REGISTRY.pop("plugin_first_coords")
