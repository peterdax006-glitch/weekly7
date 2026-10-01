import pytest
from app.money import add_cents


def test_sum():
    assert add_cents(2, 3) == 5 and add_cents(0, 0) == 0


def test_negative():
    with pytest.raises(ValueError):
        add_cents(-1, 2)


def test_type():
    with pytest.raises(TypeError):
        add_cents(1.5, 2)
