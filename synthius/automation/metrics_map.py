from synthius.metric import (
    AdvancedQualityMetrics,
    BasicQualityMetrics,
    DistanceMetrics,
    LikelihoodMetrics,
    LinkabilityMetric,
    PrivacyAgainstInference,
    PropensityScore,
    SinglingOutMetric,
)

METRICS_MAP = {
    "Utility": [
        "F1",
        "F1_Weighted",
        "F1_Macro",
        "Precision_Macro",
        "Recall_Macro",
        "Accuracy",
    ],
    "BasicQualityMetrics": [
        "Overall Quality",
        "Column Shapes",
        "Column Pair Trends",
        "Overall Diagnostic",
        "Data Validity",
        "Data Structure",
        "New Row Synthesis",
    ],
    "AdvancedQualityMetrics": [
        "Discrete KL Divergence",
        "Continuous KL Divergence",
        "CS Test",
    ],
    "LikelihoodMetrics": [
        "GM Log Likelihood",
        "BN Likelihood",
        "BN Log Likelihood",
    ],
    "DistanceMetrics": [
        "5th Percentile | DCR | R&S",
        "5th Percentile | DCR | RR",
        "5th Percentile | DCR | SS",
        "5th Percentile | NNDR | R&S",
        "5th Percentile | NNDR | RR",
        "5th Percentile | NNDR | SS",
        "Mean | DCR | RR",
        "Mean | DCR | R&S",
        "Score",
        "Removed DataPoint | R&S",
        "Removed DataPoint | RR",
        "Removed DataPoint | SS",
    ],
    "PropensityScore": [
        "Autogluon",
        "XGBoost",
        "HistGradientBoosting",
    ],
    "PrivacyAgainstInference": [
        "CategoricalKNN",
        "CategoricalNB",
        "CategoricalRF",
        "CategoricalCAP",
        "CategoricalZeroCAP",
        "CategoricalGeneralizedCAP",
        "CategoricalSVM",
        "CategoricalEnsemble",
    ],
    "LinkabilityMetric": [
        "Privacy Risk",
        "CI(95%)",
        "Main Attack Success Rate",
        "Main Attack Marginal Error ±",
        "Baseline Attack Success Rate",
        "Baseline Attack Error ±",
        "Control Attack Success Rate",
        "Control Attack Error ±",
    ],
    "SinglingOutMetric": [
        "Privacy Risk",
        "CI(95%)",
        "Main Attack Success Rate",
        "Main Attack Marginal Error ±",
        "Baseline Attack Success Rate",
        "Baseline Attack Error ±",
        "Control Attack Success Rate",
        "Control Attack Error ±",
    ],
}


METRIC_CLASS_MAP = {
    "BasicQualityMetrics": BasicQualityMetrics,
    "AdvancedQualityMetrics": AdvancedQualityMetrics,
    "LikelihoodMetrics": LikelihoodMetrics,
    "DistanceMetrics": DistanceMetrics,
    "PropensityScore": PropensityScore,
    "PrivacyAgainstInference": PrivacyAgainstInference,
    "LinkabilityMetric": LinkabilityMetric,
    "SinglingOutMetric": SinglingOutMetric,
}


METRIC_REQUIRED_PARAMS = {
    "PrivacyAgainstInference": ["key_fields", "sensitive_fields"],
    "LinkabilityMetric": [
        "linkability_n_attacks",
        "linkability_aux_cols",
        "linkability_n_neighbors",
        "control_data_path",
    ],
    "SinglingOutMetric": ["singlingout_mode", "singlingout_n_attacks", "singlingout_n_cols", "control_data_path"],
    "DistanceMetrics": ["distance_scaler", "id_column"],
}


DEFAULT_METRICS: list[str] = [
    "CategoricalZeroCAP",
    "CategoricalGeneralizedCAP",
    "CategoricalEnsemble",
    "Mean | DCR | R&S",
    "Mean | DCR | RR",
]
