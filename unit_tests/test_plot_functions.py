import pandas as pd
import numpy as np
import pytest
import sys
import os
import unittest

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.plot_functions import plot_revenue_bucket


class TestPlotRevenueBucket(unittest.TestCase):
    def test_plot_revenue_bucket(self):
        rmses = [0.1, 0.2, 0, 3]
        target = "Target"
        plot_path = "/path/to/plot.png"
        sector_colors = ["#6FAA96", "#5AC2CC", "#D1E4F2", "#2E73A9", "#003f5c"]

        result = plot_revenue_bucket(rmses, target, plot_path, sector_colors)

        self.assertTrue(os.path.exists(plot_path + f"revenue_box_plot_{target}.png"))

if __name__ == "__main__":
    unittest.main()
