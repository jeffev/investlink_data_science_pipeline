"""
Testes unitários para data_processing/feature_engineer.py.

Executa com: pytest tests/test_feature_engineer.py -v
(do diretório raiz data_science_pipeline/)
"""
import numpy as np
import pandas as pd
import pytest

from data_processing.feature_engineer import (
    winsorize,
    fill_nulls_with_sector_median,
    add_sector_zscores,
    add_composite_scores,
    engineer_features,
    INDICATOR_COLS,
    ZSCORE_COLS,
    WEIGHTS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """DataFrame mínimo com todas as colunas de indicadores preenchidas."""
    np.random.seed(seed)
    data: dict = {
        "ticker":     [f"TICK{i:02d}" for i in range(n)],
        "year":       [2022] * n,
        "sectorname": (["Financeiro"] * (n // 2)) + (["Energia"] * (n - n // 2)),
    }
    for col in INDICATOR_COLS:
        data[col] = np.random.uniform(1.0, 100.0, n)
    return pd.DataFrame(data)


# ── winsorize ─────────────────────────────────────────────────────────────────

class TestWinsorize:
    def test_clipa_valor_maximo_extremo(self):
        df = _make_df(100)
        df.loc[0, "p_l"] = 1e9
        result = winsorize(df)
        assert result["p_l"].max() < 1e9

    def test_clipa_valor_minimo_extremo(self):
        df = _make_df(100)
        df.loc[0, "p_l"] = -1e9
        result = winsorize(df)
        assert result["p_l"].min() > -1e9

    def test_nao_muta_o_input(self):
        df = _make_df(100)
        df.loc[0, "p_l"] = 1e9
        original_val = df.loc[0, "p_l"]
        winsorize(df)
        assert df.loc[0, "p_l"] == original_val

    def test_colunas_nao_indicadoras_inalteradas(self):
        df = _make_df(20)
        result = winsorize(df)
        pd.testing.assert_series_equal(result["ticker"], df["ticker"])
        pd.testing.assert_series_equal(result["sectorname"], df["sectorname"])

    def test_retorna_todas_as_colunas(self):
        df = _make_df(20)
        result = winsorize(df)
        assert set(result.columns) == set(df.columns)

    def test_sem_outliers_valores_inalterados(self):
        # Dataset uniforme — p99 e p1 são próximos dos valores; não deve clipar nada fora
        df = _make_df(50)
        result = winsorize(df, cols=["p_l"])
        # Valores entre p1 e p99 devem continuar iguais
        p1  = df["p_l"].quantile(0.01)
        p99 = df["p_l"].quantile(0.99)
        mask = (df["p_l"] >= p1) & (df["p_l"] <= p99)
        pd.testing.assert_series_equal(
            result.loc[mask, "p_l"].reset_index(drop=True),
            df.loc[mask, "p_l"].reset_index(drop=True),
        )


# ── fill_nulls_with_sector_median ─────────────────────────────────────────────

class TestFillNullsWithSectorMedian:
    def test_preenche_nan_com_mediana_do_setor(self):
        df = _make_df(6)
        df["sectorname"] = ["A", "A", "A", "B", "B", "B"]
        df["p_l"]        = [10.0, 20.0, np.nan, 50.0, 60.0, np.nan]
        result = fill_nulls_with_sector_median(df)
        assert abs(result["p_l"].iloc[2] - 15.0) < 1e-9   # mediana([10, 20]) = 15
        assert abs(result["p_l"].iloc[5] - 55.0) < 1e-9   # mediana([50, 60]) = 55

    def test_sem_nan_apos_preenchimento(self):
        df = _make_df(20)
        df.loc[0, "p_l"] = np.nan
        result = fill_nulls_with_sector_median(df)
        assert result["p_l"].isna().sum() == 0

    def test_nao_muta_o_input(self):
        df = _make_df(10)
        df.loc[0, "p_l"] = np.nan
        fill_nulls_with_sector_median(df)
        assert pd.isna(df.loc[0, "p_l"])

    def test_fallback_para_mediana_global_quando_grupo_pequeno(self):
        # Setor com só 1 linha (< MIN_SECTOR_SIZE=3): usa mediana global
        df = _make_df(6)
        df["sectorname"] = ["A", "A", "A", "SOLO", "SOLO", "SOLO"]
        df["p_l"] = [10.0, 20.0, 30.0, np.nan, np.nan, np.nan]
        # Grupos A e SOLO têm 3 linhas cada; todos NaN no SOLO
        # Mediana global de [10, 20, 30] = 20
        result = fill_nulls_with_sector_median(df)
        assert result["p_l"].isna().sum() == 0

    def test_sem_nan_originais_permanece_inalterado(self):
        df = _make_df(10)
        original_vals = df["p_l"].copy()
        result = fill_nulls_with_sector_median(df)
        pd.testing.assert_series_equal(result["p_l"], original_vals)


# ── add_sector_zscores ────────────────────────────────────────────────────────

class TestAddSectorZscores:
    def test_adiciona_colunas_z_para_cada_coluna(self):
        df = _make_df(20)
        result = add_sector_zscores(df)
        for col in ZSCORE_COLS:
            assert f"{col}_z" in result.columns

    def test_zscore_zero_quando_iqr_e_zero(self):
        # Todos valores idênticos → IQR = 0 → z-score deve ser 0
        df = _make_df(10)
        df["p_l"] = 42.0
        result = add_sector_zscores(df, cols=["p_l"])
        assert (result["p_l_z"] == 0.0).all()

    def test_nao_muta_o_input(self):
        df = _make_df(10)
        colunas_antes = set(df.columns)
        add_sector_zscores(df)
        assert set(df.columns) == colunas_antes

    def test_colunas_originais_preservadas(self):
        df = _make_df(20)
        result = add_sector_zscores(df)
        for col in ZSCORE_COLS:
            pd.testing.assert_series_equal(result[col], df[col])

    def test_z_scores_sao_numericos(self):
        df = _make_df(20)
        result = add_sector_zscores(df)
        for col in ZSCORE_COLS:
            assert pd.api.types.is_float_dtype(result[f"{col}_z"])


# ── add_composite_scores ──────────────────────────────────────────────────────

class TestAddCompositeScores:
    def test_adiciona_todas_as_colunas_de_score(self):
        df = _make_df(20)
        result = add_composite_scores(df)
        for col in ["value_score", "quality_score", "growth_score", "dividend_score", "composite_score"]:
            assert col in result.columns

    def test_scores_entre_0_e_100(self):
        df = _make_df(30)
        result = add_composite_scores(df)
        for col in ["value_score", "quality_score", "growth_score", "dividend_score", "composite_score"]:
            assert result[col].min() >= 0.0, f"{col} abaixo de 0"
            assert result[col].max() <= 100.0, f"{col} acima de 100"

    def test_composite_score_e_media_ponderada(self):
        df = _make_df(20)
        result = add_composite_scores(df)
        expected = (
            WEIGHTS["value_score"]    * result["value_score"]
            + WEIGHTS["quality_score"]  * result["quality_score"]
            + WEIGHTS["growth_score"]   * result["growth_score"]
            + WEIGHTS["dividend_score"] * result["dividend_score"]
        ).round(2)
        pd.testing.assert_series_equal(
            result["composite_score"], expected, check_names=False
        )

    def test_pesos_somam_um(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_nao_muta_o_input(self):
        df = _make_df(20)
        colunas_antes = set(df.columns)
        add_composite_scores(df)
        assert set(df.columns) == colunas_antes

    def test_sem_nan_nos_scores(self):
        df = _make_df(20)
        result = add_composite_scores(df)
        for col in ["value_score", "quality_score", "growth_score", "dividend_score", "composite_score"]:
            assert result[col].isna().sum() == 0, f"NaN encontrado em {col}"


# ── engineer_features (pipeline completo) ─────────────────────────────────────

class TestEngineerFeatures:
    def test_adiciona_colunas_z_score(self):
        df = _make_df(20)
        result = engineer_features(df)
        for col in ZSCORE_COLS:
            assert f"{col}_z" in result.columns

    def test_adiciona_composite_score(self):
        df = _make_df(20)
        result = engineer_features(df)
        assert "composite_score" in result.columns

    def test_sem_nan_nos_indicadores_apos_pipeline(self):
        df = _make_df(20)
        df.loc[0, "p_l"] = np.nan
        result = engineer_features(df)
        for col in INDICATOR_COLS:
            if col in result.columns:
                assert result[col].isna().sum() == 0, f"NaN restante em {col}"

    def test_numero_de_linhas_preservado(self):
        df = _make_df(20)
        result = engineer_features(df)
        assert len(result) == len(df)

    def test_colunas_originais_preservadas(self):
        df = _make_df(20)
        result = engineer_features(df)
        for col in df.columns:
            assert col in result.columns
