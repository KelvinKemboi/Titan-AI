from datetime import datetime, timezone

import pytest

from src.analytics.explain import explain_factor_scores
from src.data.models import FactorScore

# unit tests to check that the function correctly computes and returns explanations for factor scores based on a FactorScore object
def _make_factor_score(**overrides) -> FactorScore:
    defaults = dict(
        ticker="TEST",
        scan_run_id=42,
        computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        value_score=75.0,
        momentum_score=100.0,
        quality_score=100.0,
        solvency_score=80.0,
        volatility_score=90.0,
        composite_score=75.0 * 0.25 + 100.0 * 0.30 + 100.0 * 0.25 + 80.0 * 0.10 + 90.0 * 0.10,
        rating="STRONG BUY",
        raw_metrics={
            "Price": 100.0,
            "RSI": 65.0,
            "Trend": "Bullish",
            "Val_Metric": 1.5,
            "Val_Type": "PEG",
            "Margin": 0.25,
            "Debt": 40.0,
            "Beta": 0.9,
            "Scores": [75.0, 100.0, 100.0, 80.0, 90.0],
        },
    )
    defaults.update(overrides)
    return FactorScore(**defaults)

# ensures that the explain_factor_scores function correctly returns the metadata from the FactorScore object without modification
def test_explanation_metadata_passthrough():
    result = explain_factor_scores(_make_factor_score())

    assert result.ticker == "TEST"
    assert result.scan_run_id == 42
    assert result.rating == "STRONG BUY"
    assert result.composite_score == pytest.approx(90.75)

# correctly checks that the factors are present in the scoring order and have the correct weights as defined in the WEIGHTS dictionary
def test_factors_are_present_in_scoring_order_with_correct_weights():
    result = explain_factor_scores(_make_factor_score())

    assert [f.factor for f in result.factors] == [
        "value", "momentum", "quality", "solvency", "volatility",
    ]
    weights = {f.factor: f.weight for f in result.factors}
    assert weights == {
        "value": 0.25, "momentum": 0.30, "quality": 0.25,
        "solvency": 0.10, "volatility": 0.10,
    }

# checks that the contributions for each factor are correctly calculated as score * weight and that their sum equals the composite score
def test_contributions_are_score_times_weight_and_sum_to_composite():
    result = explain_factor_scores(_make_factor_score())

    by_factor = {f.factor: f for f in result.factors}
    assert by_factor["value"].contribution == pytest.approx(75.0 * 0.25)
    assert by_factor["momentum"].contribution == pytest.approx(100.0 * 0.30)
    assert by_factor["quality"].contribution == pytest.approx(100.0 * 0.25)
    assert by_factor["solvency"].contribution == pytest.approx(80.0 * 0.10)
    assert by_factor["volatility"].contribution == pytest.approx(90.0 * 0.10)

    assert sum(f.contribution for f in result.factors) == pytest.approx(result.composite_score)

# checks that the driver text for each factor is correctly generated based on the raw_metrics provided in the FactorScore object
def test_driver_text_reuses_analyst_thresholds_with_actual_values():
    result = explain_factor_scores(_make_factor_score())
    by_factor = {f.factor: f for f in result.factors}

    assert by_factor["value"].driver == (
        "PEG of 1.50 (PEG < 1.0 is elite, PEG > 3.0 is poor)"
    )
    assert by_factor["momentum"].driver == (
        "RSI 65.0, Bullish trend (RSI 40-75 is the target range)"
    )
    assert by_factor["quality"].driver == (
        "Net margins of 25.0% (>20% margins is elite)"
    )
    assert by_factor["solvency"].driver == (
        "Debt/Equity of 40.0 (<50% is elite)"
    )
    assert by_factor["volatility"].driver == (
        "Beta of 0.90 (<1.0 is considered safe)"
    )

# checks that when the Val_Type is "Unknown", the driver text for the value factor correctly indicates that no PEG or P/E is available and defaults to an assumed-expensive Val_Metric
def test_value_driver_when_val_type_unknown():
    factor_score = _make_factor_score(
        raw_metrics={
            "Price": 50.0, "RSI": 50.0, "Trend": "Bearish",
            "Val_Metric": 5.0, "Val_Type": "Unknown",
            "Margin": 0.05, "Debt": 150.0, "Beta": 1.5,
            "Scores": [0.0, 50.0, 25.0, 25.0, 30.0],
        },
    )
    result = explain_factor_scores(factor_score)
    value_driver = next(f.driver for f in result.factors if f.factor == "value")

    assert value_driver == (
        "No PEG or P/E available - defaulted to an assumed-expensive "
        "5.00 (PEG < 1.0 is elite, PEG > 3.0 is poor)"
    )

# check that if the raw_metrics field is missing a required field, the function does not crash and instead returns a driver message indicating that the field is unavailable
def test_missing_raw_metrics_field_degrades_gracefully_instead_of_crashing():
    factor_score = _make_factor_score(raw_metrics={})
    result = explain_factor_scores(factor_score)

    drivers = {f.factor: f.driver for f in result.factors}
    assert drivers["value"] == "Val_Metric unavailable"
    assert drivers["momentum"] == "Unknown trend"
    assert drivers["quality"] == "Margin unavailable"
    assert drivers["solvency"] == "Debt/Equity unavailable"
    assert drivers["volatility"] == "Beta unavailable"


# check that if the raw_metrics field is None, the function does not crash and still returns a FactorScoreExplanation object with 5 factors
def test_null_raw_metrics_does_not_crash():
    factor_score = _make_factor_score(raw_metrics=None)
    result = explain_factor_scores(factor_score)

    assert len(result.factors) == 5

# check that the output of explain_factor_scores is structured data and not prose, ensuring that the driver field for each factor does not contain newline characters
def test_output_is_structured_data_not_prose():
    result = explain_factor_scores(_make_factor_score())

    for factor in result.factors:
        assert isinstance(factor.score, float)
        assert isinstance(factor.weight, float)
        assert isinstance(factor.contribution, float)
        assert isinstance(factor.driver, str)
        assert "\n" not in factor.driver
