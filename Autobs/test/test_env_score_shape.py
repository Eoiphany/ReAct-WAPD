"""注释
命令:
1. `Python -B -m unittest Autobs.test.test_env_score_shape`

参数含义:
- `Autobs.test.test_env_score_shape`: 验证 score 在 coverage 达到保守下界后，会把主导优化方向切向频谱效率，
  同时保留较小的 coverage 余量奖励。
"""

from __future__ import annotations

import unittest

from Autobs.env.utils import compute_score_components


class EnvScoreShapeTests(unittest.TestCase):
    def test_score_prefers_higher_coverage_when_tradeoff_exists(self) -> None:
        high_coverage_low_se = compute_score_components(
            coverage=0.62,
            spectral_efficiency=0.35,
            rss_margin=0.20,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )
        lower_coverage_high_se = compute_score_components(
            coverage=0.48,
            spectral_efficiency=1.60,
            rss_margin=0.20,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )

        self.assertGreater(high_coverage_low_se["score"], lower_coverage_high_se["score"])

    def test_score_prefers_higher_se_after_coverage_floor_is_met(self) -> None:
        higher_coverage_lower_se = compute_score_components(
            coverage=0.66,
            spectral_efficiency=0.40,
            rss_margin=0.20,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )
        slightly_lower_coverage_higher_se = compute_score_components(
            coverage=0.64,
            spectral_efficiency=0.90,
            rss_margin=0.20,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )

        self.assertGreater(
            slightly_lower_coverage_higher_se["score"],
            higher_coverage_lower_se["score"],
        )

    def test_score_keeps_small_margin_bonus_at_same_coverage(self) -> None:
        low_margin = compute_score_components(
            coverage=0.65,
            spectral_efficiency=0.40,
            rss_margin=0.05,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )
        high_margin = compute_score_components(
            coverage=0.65,
            spectral_efficiency=0.40,
            rss_margin=0.35,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )

        self.assertGreater(high_margin["margin_term"], low_margin["margin_term"])
        self.assertGreater(high_margin["score"], low_margin["score"])
        self.assertLess(high_margin["margin_term"], high_margin["coverage_term"])

    def test_score_penalizes_low_se_at_same_coverage_and_margin(self) -> None:
        high_se = compute_score_components(
            coverage=0.66,
            spectral_efficiency=0.85,
            rss_margin=0.20,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )
        low_se = compute_score_components(
            coverage=0.66,
            spectral_efficiency=0.20,
            rss_margin=0.20,
            coverage_target=0.65,
            spectral_efficiency_target=0.80,
        )

        self.assertGreater(high_se["se_term"], low_se["se_term"])
        self.assertGreater(high_se["score"], low_se["score"])


if __name__ == "__main__":
    unittest.main()
