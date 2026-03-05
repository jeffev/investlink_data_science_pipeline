import math
from sqlalchemy import (
    Column, String, Integer, Float, Date, DateTime,
    ForeignKey, UniqueConstraint, text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class StockIndicatorHistory(Base):
    """
    Historical financial indicators per ticker per year,
    scraped from StatusInvest (2008 onwards).
    """
    __tablename__ = "stock_indicators_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("stocks.ticker"), nullable=False, index=True)
    year = Column(Integer, nullable=False)

    # --- Valuation ---
    dy = Column(Float)               # Dividend Yield (%)
    p_l = Column(Float)              # P/L — Price-to-Earnings
    peg_ratio = Column(Float)        # PEG Ratio
    p_vp = Column(Float)             # P/VP — Price-to-Book
    ev_ebitda = Column(Float)        # EV/EBITDA
    ev_ebit = Column(Float)          # EV/EBIT
    p_ebitda = Column(Float)         # P/EBITDA
    p_ebit = Column(Float)           # P/EBIT
    vpa = Column(Float)              # VPA — Book Value Per Share
    p_ativo = Column(Float)          # P/Ativo
    lpa = Column(Float)              # LPA — Earnings Per Share
    p_sr = Column(Float)             # P/SR — Price-to-Sales
    p_cap_giro = Column(Float)       # P/Cap. Giro
    p_ativo_circ_liq = Column(Float) # P/Ativo Circ. Líq.

    # --- Leverage ---
    div_liq_pl = Column(Float)       # Dív. Líquida/PL
    div_liq_ebitda = Column(Float)   # Dív. Líquida/EBITDA
    div_liq_ebit = Column(Float)     # Dív. Líquida/EBIT
    pl_ativos = Column(Float)        # PL/Ativos
    passivos_ativos = Column(Float)  # Passivos/Ativos

    # --- Profitability ---
    m_bruta = Column(Float)          # Margem Bruta (%)
    m_ebitda = Column(Float)         # Margem EBITDA (%)
    m_ebit = Column(Float)           # Margem EBIT (%)
    m_liquida = Column(Float)        # Margem Líquida (%)
    roe = Column(Float)              # ROE (%)
    roa = Column(Float)              # ROA (%)
    roic = Column(Float)             # ROIC (%)

    # --- Efficiency & Liquidity ---
    giro_ativos = Column(Float)      # Giro Ativos
    liq_corrente = Column(Float)     # Liquidez Corrente

    # --- Growth ---
    cagr_receitas_5 = Column(Float)  # CAGR Receitas 5 Anos (%)
    cagr_lucros_5 = Column(Float)    # CAGR Lucros 5 Anos (%)

    # --- Derived (calculated during scraping) ---
    graham_formula = Column(Float)   # √(22.5 × LPA × VPA)

    scraped_at = Column(DateTime, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("ticker", "year", name="uq_indicator_ticker_year"),
    )

    @staticmethod
    def calc_graham(lpa, vpa):
        if lpa is not None and vpa is not None and lpa >= 0 and vpa >= 0:
            return round(math.sqrt(22.5 * lpa * vpa), 2)
        return None


class StockPriceHistory(Base):
    """
    Weekly historical closing prices per ticker, from Yahoo Finance (.SA suffix).
    """
    __tablename__ = "stock_prices_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("stocks.ticker"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    close_price = Column(Float)
    open_price = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
    )


class StockPrediction(Base):
    """
    ML model predictions per ticker, refreshed each pipeline run.
    """
    __tablename__ = "stock_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("stocks.ticker"), nullable=False, index=True)
    run_date = Column(DateTime, nullable=False, server_default=text("NOW()"))
    label = Column(String(10))        # BARATA / NEUTRA / CARA
    prob_barata = Column(Float)
    prob_neutra = Column(Float)
    prob_cara = Column(Float)
    composite_score = Column(Float)   # 0–100, quanto maior melhor
    model_version = Column(String(50))

    __table_args__ = (
        UniqueConstraint("ticker", "run_date", name="uq_prediction_ticker_rundate"),
    )
