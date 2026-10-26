import unittest

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import charts, config


class TestCharts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(rng.integers(0, 10, size=(200, 20)), columns=config.FEATURES)
        self.df[config.TARGET] = rng.uniform(0.3, 0.7, 200)
        self.pred = self.df[config.TARGET] + rng.normal(0, 0.02, 200)

    def test_all_charts_build(self):
        imp = pd.DataFrame({"feature": config.FEATURES, "importance_mean": 0.01, "importance_std": 0.001})
        models = {"A": {"MAE": 1, "RMSE": 2, "R2": 0.5}, "B": {"MAE": 2, "RMSE": 3, "R2": 0.1}}
        figs = [charts.plot_target_distribution(self.df[config.TARGET]),
                charts.plot_feature_distributions(self.df),
                charts.plot_correlation_heatmap(self.df.corr()),
                charts.plot_metric_comparison(models),
                charts.plot_actual_vs_predicted(self.df[config.TARGET], self.pred),
                charts.plot_residuals(self.df[config.TARGET], self.pred),
                charts.plot_feature_importance(imp),
                charts.plot_feature_contributions(pd.Series(np.linspace(-1, 1, 20), index=config.FEATURES))]
        for fig in figs:
            self.assertIsInstance(fig, Figure)


if __name__ == "__main__":
    unittest.main()
