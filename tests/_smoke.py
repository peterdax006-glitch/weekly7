import time, numpy as np, pandas as pd
from engine import parity as P, parity_suite as S
from tests import test_parity as T
inp = T.inp.__wrapped__() if hasattr(T.inp,'__wrapped__') else None
