import unittest

from summarize_e7_lessismore_projection import projected_ms


class LessIsMoreProjectionTest(unittest.TestCase):
    def test_four_dense_and_twenty_eight_sparse_layers(self):
        self.assertAlmostEqual(projected_ms(10, 1, 2000, 500), 33.0)


if __name__ == "__main__":
    unittest.main()
