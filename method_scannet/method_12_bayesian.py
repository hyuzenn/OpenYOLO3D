"""METHOD_12 — Bayesian probability accumulation registration gate.

Phase 2 of the registration axis. Each instance carries a posterior
P(real | history). A visible-this-frame observation pushes the posterior
toward 1.0; a not-visible-this-frame observation pulls it toward 0.0.
The instance is "confirmed" once the posterior exceeds ``threshold``.

Forward-only state; no offline knowledge required.
"""
from __future__ import annotations

from collections import defaultdict


class BayesianGate:
    """Posterior-based gate.

    Args:
        prior: P(real) before any observation. Default 0.5.
        detection_likelihood: P(visible | real). Default 0.8.
        false_positive_rate: P(visible | not real). Default 0.2.
        threshold: posterior >= threshold confirms the instance.
            Default 0.95.
    """

    def __init__(
        self,
        prior: float = 0.5,
        detection_likelihood: float = 0.8,
        false_positive_rate: float = 0.2,
        threshold: float = 0.95,
    ) -> None:
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        if not 0.0 < detection_likelihood < 1.0:
            raise ValueError(
                f"detection_likelihood must be in (0, 1), got {detection_likelihood}"
            )
        self.prior = float(prior)
        self.likelihood = float(detection_likelihood)
        self.fpr = float(false_positive_rate)
        self.threshold = float(threshold)
        self._posteriors: dict[int, float] = defaultdict(lambda: self.prior)
        self._confirmed: set[int] = set()

    def reset(self) -> None:
        self._posteriors.clear()
        self._confirmed.clear()

    def _update(self, p: float, observed_visible: bool) -> float:
        """Single Bayesian update."""
        if observed_visible:
            num = self.likelihood * p
            denom = self.likelihood * p + self.fpr * (1.0 - p)
        else:
            num = (1.0 - self.likelihood) * p
            denom = (1.0 - self.likelihood) * p + (1.0 - self.fpr) * (1.0 - p)
        return num / max(denom, 1e-12)

    def gate(self, visible_instances) -> list[int]:
        seen_this_frame = {int(i) for i in visible_instances}
        # Only update the posteriors of instances we've seen at least once.
        # (Unseen-forever instances stay at the prior and never grow.)
        for k in seen_this_frame:
            self._posteriors[k] = self._update(self._posteriors[k], True)
            if self._posteriors[k] >= self.threshold:
                self._confirmed.add(k)
        return sorted(seen_this_frame & self._confirmed)

    def posterior(self, instance_id: int) -> float:
        return float(self._posteriors[int(instance_id)])

    @property
    def confirmed_count(self) -> int:
        return len(self._confirmed)
