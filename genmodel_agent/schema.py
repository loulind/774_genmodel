"""Typed records passed between the agent's steps.

Every object here is JSON-serialisable, because each one is shown to the LLM,
written to the run directory, and quoted in the decision log. Fields are
measurements, not judgements: the profile says "imbalance ratio 27.3", never
"imbalanced". Turning numbers into decisions is the planner's job.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["image", "tabular", "counts"]


class LabelSummary(BaseModel):
    n_classes: int
    class_names: list[str]
    class_counts: dict[str, int]
    imbalance_ratio: float              # largest class / smallest class
    minority_class: str
    minority_count: int
    minority_fraction: float


class MissingSummary(BaseModel):
    overall_fraction: float             # missing cells / all cells
    n_features_with_missing: int
    max_feature_fraction: float
    rows_with_any_missing_fraction: float


class CountSummary(BaseModel):
    """Statistics that decide between Gaussian, Poisson, NB and ZINB likelihoods."""
    mean_library_size: float
    median_library_size: float
    fraction_overdispersed_features: float   # var > mean, among expressed features
    median_dispersion_ratio: float           # median var/mean
    observed_zero_fraction: float
    poisson_expected_zero_fraction: float
    nb_expected_zero_fraction: float
    excess_zeros_vs_nb: float                # observed - NB-expected (>0 suggests ZI)
    n_features_assessed: int                 # genes detected in >= 5% of cells
    # Per-gene MLE comparison (moment matching cannot separate ZI from overdispersion)
    fraction_genes_zinb_preferred: float | None = None   # ZINB lower AIC than NB
    median_zero_inflation_pi: float | None = None


class ShapeSummary(BaseModel):
    """Distribution shape across continuous features."""
    median_abs_skew: float | None = None
    fraction_heavily_skewed: float | None = None     # |skew| > 1
    fraction_multimodal: float | None = None         # bimodality coefficient > 0.555
    median_excess_kurtosis: float | None = None


class DataProfile(BaseModel):
    name: str
    source: str
    container: str                      # npz | h5ad | csv | npy
    modality: Modality
    modality_evidence: str

    n_samples: int
    n_features: int                     # variables (columns / pixels / genes)
    n_encoded_features: int             # after one-hot encoding categoricals
    n_over_p: float
    image_shape: list[int] | None = None   # [H, W, C]

    n_continuous: int = 0
    n_discrete: int = 0                 # integer-valued with few levels
    n_binary: int = 0
    n_categorical: int = 0              # non-numeric
    categorical_cardinalities: dict[str, int] = Field(default_factory=dict)

    value_min: float | None = None
    value_max: float | None = None
    is_integer_valued: bool = False
    is_nonnegative: bool = False
    zero_fraction: float = 0.0
    is_sparse_storage: bool = False

    labels: LabelSummary | None = None
    missing: MissingSummary
    counts: CountSummary | None = None
    shape: ShapeSummary = Field(default_factory=ShapeSummary)

    intrinsic_dimension: float | None = None
    pca_components_90: int | None = None     # components for 90% variance
    duplicate_fraction: float | None = None        # exact duplicate rows
    near_duplicate_fraction: float | None = None   # NN distance < 0.2 x median NN distance
    predefined_splits: dict[str, int] | None = None
    dense_memory_mb: float

    def brief(self) -> str:
        """One-line description for logs and the robustness table."""
        parts = [f"{self.modality}", f"n={self.n_samples:,}", f"p={self.n_features:,}"]
        if self.labels:
            parts.append(f"{self.labels.n_classes} classes (imb {self.labels.imbalance_ratio:.1f})")
        if self.missing.overall_fraction > 0:
            parts.append(f"{100 * self.missing.overall_fraction:.1f}% missing")
        if self.zero_fraction > 0.5:
            parts.append(f"{100 * self.zero_fraction:.0f}% zeros")
        return ", ".join(parts)
