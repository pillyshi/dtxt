from collections.abc import Iterator

import pytest

from dtxt._config import reset


@pytest.fixture(autouse=True)
def _reset_global_config() -> Iterator[None]:
    reset()
    yield
    reset()
