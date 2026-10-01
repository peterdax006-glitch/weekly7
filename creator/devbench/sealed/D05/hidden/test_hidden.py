import time
from app.primes import primes_below

def test_edges():
    assert primes_below(0) == [] and primes_below(2) == [] and primes_below(3) == [2]

def test_fast_and_correct():
    t0 = time.perf_counter()
    ps = primes_below(300000)
    assert time.perf_counter() - t0 < 1.0
    assert len(ps) == 25997 and ps[-1] == 299993
