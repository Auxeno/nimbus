"""Test configuration for nimbus tests."""

import os
import pytest


def pytest_generate_tests(metafunc):
    """Run each test with JIT enabled and disabled."""
    if "jit_mode" in metafunc.fixturenames:
        metafunc.parametrize("jit_mode", ["no_jit", "jit"])


@pytest.fixture
def jit_mode(request):
    """Set JAX JIT compilation mode."""
    if request.param == "no_jit":
        os.environ["JAX_DISABLE_JIT"] = "1"
    else:
        os.environ["JAX_DISABLE_JIT"] = "0"
    return request.param
