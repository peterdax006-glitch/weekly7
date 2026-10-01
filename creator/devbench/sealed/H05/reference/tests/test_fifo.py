import pytest
from app.fifo import Queue


def test_order_and_len():
    q = Queue()
    q.push(1); q.push(2)
    assert len(q) == 2 and q.pop() == 1 and len(q) == 1 and q.pop() == 2 and len(q) == 0


def test_empty():
    with pytest.raises(IndexError):
        Queue().pop()
