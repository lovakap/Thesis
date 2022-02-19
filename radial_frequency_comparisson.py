from src.aspire.basis.fb_2d import FBBasis2D
from tests.test_FBbasis2D import FBBasis2DTestCase
import numpy as np

basis = FBBasis2D((8, 8), dtype=np.float32)
indices = basis.indices()

fb = FBBasis2DTestCase()
fb.testFBBasis2DEvaluate()