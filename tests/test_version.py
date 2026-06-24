from __future__ import annotations

import vlmscope


def test_version_is_exposed() -> None:
    assert isinstance(vlmscope.__version__, str)
    # semantic version with at least major.minor.patch
    assert vlmscope.__version__.count(".") >= 2
