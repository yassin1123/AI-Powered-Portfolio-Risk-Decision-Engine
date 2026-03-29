"""Load config.yaml merged with environment (python-dotenv + pydantic-settings)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnomalyConfig(BaseModel):
    z_window: int = 60
    cusum_k_sigma: float = 0.5
    cusum_h_sigma: float = 4.0
    mahalanobis_chi2_quantile: float = 0.999
    drawdown_watch: float = 0.05
    drawdown_warning: float = 0.10
    drawdown_critical: float = 0.20
    conjunction_min_layers: int = 2


class RegimeConfig(BaseModel):
    vol_elevated_garch_mult: float = 1.25
    stressed_tail_mult: float = 1.35
    stressed_avg_corr: float = 0.55
    transition_anomaly_count: int = 4
    hmm_min_history: int = 60
    use_hmm_features: bool = False


class CorrelationRegimeSignalConfig(BaseModel):
    rolling_window: int = 120
    z_high: float = 1.5
    z_low: float = -1.0
    eps_std: float = 1e-6


class DecisionEngineConfig(BaseModel):
    stress_corr_z_threshold: float = 1.0
    anomaly_suppress_count: int = 3
    exposure_scale_stress: float = 0.55
    exposure_scale_transition: float = 0.82
    confidence_anomaly_divisor: float = Field(
        default=14.0,
        ge=1e-6,
        description="noise = clip(anomaly_count/divisor, 0, 1); larger divisor → milder confidence penalty.",
    )
    confidence_transition_penalty: float = Field(
        default=0.08,
        ge=0.0,
        description="Subtracted from raw trace confidence when priority is transition.",
    )


class PortfolioConfig(BaseModel):
    target_ann_vol: float = 0.10
    max_gross_leverage: float = 1.0
    max_single_weight: float = 0.35
    min_cash_weight: float = 0.0
    turnover_cap: float = 0.5
    cost_bps: float = 10.0
    long_only: bool = True


class BacktestConfig(BaseModel):
    fill_rule: str = "next_bar"
    initial_cash: float = 1_000_000.0
    walkforward_train_bars: int = 504
    walkforward_test_bars: int = 63
    walkforward_expanding: bool = False
    var_mc_sims: int = Field(
        default=1500,
        description="MC paths per bar for backtest VaR (live loop uses mc_sims).",
    )
    # Synthetic price path for `backtest.run` default panel: without drift, EW Sharpe ≈ 0 minus rf drag.
    synthetic_daily_drift: float = Field(
        default=0.00028,
        description="Mean log-return per bar added to each synthetic asset (~7% annualized at 252d).",
    )
    synthetic_vol_per_bar: float = Field(
        default=0.01,
        description="Std dev of per-bar log shock multiplier (Gaussian) in synthetic panel.",
    )
    rebalance_every_bars: int = Field(
        default=1,
        ge=1,
        description="1 = daily rebalance; 5 ≈ weekly; 21 ≈ monthly.",
    )
    risk_disagreement_rel_threshold: float = Field(
        default=0.35,
        description="|HS VaR − MC VaR| / max(MC, ε) above this → risk_disagreement flag in logs.",
    )


class AlphaConfig(BaseModel):
    mom_windows: list[int] = Field(default_factory=lambda: [20, 60, 120])
    """Optional positive weights per mom_windows entry; if empty, horizons are equally weighted."""
    mom_window_weights: list[float] = Field(default_factory=list)
    mom_ewma_lambda: float = Field(
        default=0.94,
        description="EWMA decay for per-horizon vol scaling (RiskMetrics-style, λ).",
    )
    mom_skip_recent_bars: int = Field(
        default=0,
        description="Exclude the last N daily log-return rows before building signals (0 = use all history).",
    )
    mom_vol_floor_quantile: float = Field(
        default=0.1,
        description="Cross-sectional quantile of EWMA vol used as a floor to stabilize z-scores.",
    )
    mr_window: int = 5
    mr_slow_window: int = Field(
        default=63,
        description="Slow window (bars) for mean daily return anchor in residual mean-reversion.",
    )
    mr_use_residual: bool = Field(
        default=True,
        description="If True, MR targets short-horizon deviation from slow drift; else legacy -z(short sum).",
    )
    xsec_quantile: float = 0.2
    xsec_use_rank_gaussian: bool = Field(
        default=True,
        description="If True, map momentum ranks to Gaussian scores; else legacy quantile long/short flags.",
    )
    carry_level_weight: float = 0.65
    carry_change_weight: float = 0.35
    carry_change_bars: int = 21
    carry_tanh_scale: float = 3.0
    normalize_cross_section: bool = Field(
        default=True,
        description="Z-score combined sleeve scores across names before correlation tilt.",
    )
    regime_dynamic_weights: bool = Field(
        default=True,
        description="Scale momentum/MR/xsec/carry weights by correlation regime bucket.",
    )
    combine_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "momentum": 0.28,
            "mean_reversion": 0.18,
            "cross_sectional": 0.22,
            "carry": 0.12,
            "correlation_regime": 0.20,
        }
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=False, alias="json")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    alpha_vantage_api_key: str = Field(default="", validation_alias="ALPHA_VANTAGE_API_KEY")

    universe_profile: str = Field(
        default="core",
        description="full | core — core omits vol ETPs/crypto for stable GARCH/VaR in the UI",
    )

    lookback_days: int = 504
    history_start: str = Field(
        default="2010-01-01",
        description="ISO date: yfinance history start (elite brief §4.1 / scripts/build_data_panel.py).",
    )
    poll_interval_live_sec: int = 60
    poll_interval_sim_sec: int = 1
    simulation_mode: bool = True
    simulation_speed: float = 1.0
    simulation_noise_std: float = 0.0

    mc_sims: int = 10000
    var_confidence_levels: list[float] = Field(default_factory=lambda: [0.95, 0.99])
    var_horizons_days: list[int] = Field(default_factory=lambda: [1, 10])

    risk_free_annual: float = 0.035
    beta_window: int = 252
    garch_window: int = 504
    garch_refit_days: int = 7
    dcc_refit_days: int = 1

    covariance_window: int = 252
    cholesky_frobenius_threshold: float = 0.05

    portfolio_total_value: float = 1_000_000.0
    risk_limit_var_99: float = 0.05
    risk_budget_default: float = 0.05
    weight_drift_threshold: float = 0.05
    tail_multiplier_hedge: float = 1.5
    avg_corr_alert: float = 0.7
    drawdown_hard_stop: float = 0.15

    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    correlation_signal: CorrelationRegimeSignalConfig = Field(
        default_factory=CorrelationRegimeSignalConfig
    )
    decision_engine: DecisionEngineConfig = Field(default_factory=DecisionEngineConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    alpha: AlphaConfig = Field(default_factory=AlphaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    dash_host: str = "127.0.0.1"
    dash_port: int = 8050
    dash_interval_ms: int = 500

    config_path: Path = Field(default=Path("config.yaml"), exclude=True)


def load_settings(config_path: Path | None = None) -> AppSettings:
    load_dotenv()
    path = config_path or Path("config.yaml")
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return AppSettings(**data, config_path=path)
