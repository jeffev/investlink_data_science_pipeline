"""
Testes unitários para data_processing/labeler.py.

Executa com: pytest tests/test_labeler.py -v
(do diretório raiz data_science_pipeline/)
"""
import numpy as np
import pandas as pd
import pytest

from data_processing.labeler import add_labels, drop_unlabeled


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(price_current: list, price_next: list) -> pd.DataFrame:
    """Monta DataFrame mínimo para os testes de labeling."""
    n = len(price_current)
    return pd.DataFrame({
        "ticker":        [f"TICK{i}" for i in range(n)],
        "year":          list(range(2018, 2018 + n)),
        "sectorname":    ["Financeiro"] * n,
        "price_current": price_current,
        "price_next":    price_next,
    })


# ── add_labels (use_relative=False) ──────────────────────────────────────────

class TestAddLabelsAbsolute:
    """Testa labeling com retorno absoluto (sem ajuste Ibovespa)."""

    def test_barata_quando_retorno_acima_do_threshold(self):
        df = _make_df([100.0], [120.0])   # retorno = +20% > +15%
        result = add_labels(df, use_relative=False)
        assert result["label"].iloc[0] == "BARATA"

    def test_cara_quando_retorno_abaixo_do_threshold(self):
        df = _make_df([100.0], [80.0])    # retorno = -20% < -15%
        result = add_labels(df, use_relative=False)
        assert result["label"].iloc[0] == "CARA"

    def test_neutra_quando_retorno_dentro_do_range(self):
        df = _make_df([100.0], [105.0])   # retorno = +5%, dentro de ±15%
        result = add_labels(df, use_relative=False)
        assert result["label"].iloc[0] == "NEUTRA"

    def test_exatamente_no_threshold_positivo_vira_neutra(self):
        # alpha == threshold → não é estritamente maior → NEUTRA
        df = _make_df([100.0], [115.0])   # retorno = +15.0%
        result = add_labels(df, use_relative=False)
        assert result["label"].iloc[0] == "NEUTRA"

    def test_stock_return_calculado_corretamente(self):
        df = _make_df([100.0], [130.0])
        result = add_labels(df, use_relative=False)
        assert abs(result["stock_return"].iloc[0] - 0.30) < 1e-9

    def test_multiplas_linhas_labels_corretos(self):
        df = _make_df(
            [100.0, 100.0, 100.0],
            [120.0, 105.0,  80.0],
        )
        result = add_labels(df, use_relative=False)
        assert list(result["label"]) == ["BARATA", "NEUTRA", "CARA"]

    def test_threshold_customizado_gain(self):
        df = _make_df([100.0], [108.0])   # retorno = +8%
        # Com threshold 5%: BARATA; com threshold 10%: NEUTRA
        assert add_labels(df, use_relative=False, gain_threshold=0.05)["label"].iloc[0] == "BARATA"
        assert add_labels(df, use_relative=False, gain_threshold=0.10)["label"].iloc[0] == "NEUTRA"

    def test_threshold_customizado_loss(self):
        df = _make_df([100.0], [93.0])    # retorno = -7%
        assert add_labels(df, use_relative=False, loss_threshold=0.05)["label"].iloc[0] == "CARA"
        assert add_labels(df, use_relative=False, loss_threshold=0.10)["label"].iloc[0] == "NEUTRA"

    def test_input_nao_mutado(self):
        df = _make_df([100.0], [120.0])
        colunas_originais = set(df.columns)
        add_labels(df, use_relative=False)
        assert set(df.columns) == colunas_originais  # df original não ganhou novas colunas

    def test_alpha_igual_a_stock_return_sem_ibov(self):
        df = _make_df([100.0], [120.0])
        result = add_labels(df, use_relative=False)
        assert abs(result["alpha"].iloc[0] - result["stock_return"].iloc[0]) < 1e-9


# ── add_labels — linhas sem preço ────────────────────────────────────────────

class TestAddLabelsPrecoInvalido:
    def test_label_none_quando_price_current_nulo(self):
        df = _make_df([np.nan], [120.0])
        result = add_labels(df, use_relative=False)
        assert result["label"].iloc[0] is None or pd.isna(result["label"].iloc[0])

    def test_label_none_quando_price_next_nulo(self):
        df = _make_df([100.0], [np.nan])
        result = add_labels(df, use_relative=False)
        assert result["label"].iloc[0] is None or pd.isna(result["label"].iloc[0])

    def test_label_none_quando_price_current_zero(self):
        df = _make_df([0.0], [100.0])
        result = add_labels(df, use_relative=False)
        assert result["label"].iloc[0] is None or pd.isna(result["label"].iloc[0])

    def test_df_vazio_sem_erro(self):
        df = _make_df([], [])
        result = add_labels(df, use_relative=False)
        assert len(result) == 0


# ── drop_unlabeled ────────────────────────────────────────────────────────────

class TestDropUnlabeled:
    def test_remove_linhas_sem_label(self):
        df = _make_df([np.nan, 100.0], [120.0, 120.0])
        labeled = add_labels(df, use_relative=False)
        result = drop_unlabeled(labeled)
        assert len(result) == 1
        assert result["label"].iloc[0] == "BARATA"

    def test_df_vazio_quando_tudo_sem_label(self):
        df = _make_df([np.nan, np.nan], [120.0, 130.0])
        labeled = add_labels(df, use_relative=False)
        result = drop_unlabeled(labeled)
        assert len(result) == 0

    def test_index_resetado(self):
        df = _make_df([np.nan, 100.0, 100.0], [120.0, 130.0, 80.0])
        labeled = add_labels(df, use_relative=False)
        result = drop_unlabeled(labeled)
        assert list(result.index) == [0, 1]

    def test_nao_remove_nada_quando_tudo_rotulado(self):
        df = _make_df([100.0, 100.0], [120.0, 80.0])
        labeled = add_labels(df, use_relative=False)
        result = drop_unlabeled(labeled)
        assert len(result) == 2
