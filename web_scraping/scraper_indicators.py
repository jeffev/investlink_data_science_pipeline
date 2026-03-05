"""
Scrapes historical financial indicators from StatusInvest for a given stock ticker.

Adapted from analise-mercado-financeiro-brasil/src/scraping/scraper.py.
Key differences:
  - Returns structured dict instead of writing to CSV.
  - Supports headless Chrome.
  - Calculates Graham formula during scraping.
  - Explicit mapping from indicator display names to DB column names.
"""
import math
import time
import logging
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)

# Maps StatusInvest display names → DB column names in StockIndicatorHistory
INDICATOR_MAP = {
    "D.Y": "dy",
    "P/L": "p_l",
    "PEG RATIO": "peg_ratio",
    "P/VP": "p_vp",
    "EV/EBITDA": "ev_ebitda",
    "EV/EBIT": "ev_ebit",
    "P/EBITDA": "p_ebitda",
    "P/EBIT": "p_ebit",
    "VPA": "vpa",
    "P/ATIVO": "p_ativo",
    "LPA": "lpa",
    "P/SR": "p_sr",
    "P/CAP. GIRO": "p_cap_giro",
    "P/ATIVO CIRC. LIQ.": "p_ativo_circ_liq",
    "DÍV. LÍQUIDA/PL": "div_liq_pl",
    "DÍV. LÍQUIDA/EBITDA": "div_liq_ebitda",
    "DÍV. LÍQUIDA/EBIT": "div_liq_ebit",
    "PL/ATIVOS": "pl_ativos",
    "PASSIVOS/ATIVOS": "passivos_ativos",
    "M. BRUTA": "m_bruta",
    "M. EBITDA": "m_ebitda",
    "M. EBIT": "m_ebit",
    "M. LÍQUIDA": "m_liquida",
    "ROE": "roe",
    "ROA": "roa",
    "ROIC": "roic",
    "GIRO ATIVOS": "giro_ativos",
    "LIQ. CORRENTE": "liq_corrente",
    "CAGR RECEITAS 5 ANOS": "cagr_receitas_5",
    "CAGR LUCROS 5 ANOS": "cagr_lucros_5",
}


def _parse_float(raw: str) -> float | None:
    """Converts a scraped string like '11,75%' or '-0,13' to float. Returns None if invalid."""
    if not raw:
        return None
    cleaned = raw.strip().replace(",", ".").rstrip("%")
    if cleaned in ("-", "—", ""):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_year(text: str, current_year: int) -> int | None:
    """Maps 'ATUAL' to current_year; otherwise parses the year string as int."""
    text = text.strip()
    if text.upper() == "ATUAL":
        return current_year
    try:
        return int(text)
    except ValueError:
        return None


def _calc_graham(lpa: float | None, vpa: float | None) -> float | None:
    if lpa is not None and vpa is not None and lpa >= 0 and vpa >= 0:
        return round(math.sqrt(22.5 * lpa * vpa), 2)
    return None


def _make_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    return webdriver.Chrome(options=options)


def _close_popups(driver) -> None:
    for popup in driver.find_elements(By.CSS_SELECTOR, ".popup-fixed .btn-close"):
        try:
            WebDriverWait(driver, 3).until(EC.element_to_be_clickable(popup)).click()
            time.sleep(0.5)
        except Exception:
            pass


def scrape_indicators(ticker: str, headless: bool = True) -> dict[int, dict]:
    """
    Opens the StatusInvest page for `ticker`, navigates to the historical
    indicators table and returns all available years.

    Returns:
        dict mapping year (int) → dict of {column_name: float | None}
        Includes 'graham_formula' calculated from lpa + vpa.
        Returns {} on failure.

    Example:
        {
            2023: {"dy": 7.87, "p_l": 7.37, "roic": 45.2, "graham_formula": 28.4, ...},
            2022: {...},
        }
    """
    driver = _make_driver(headless=headless)
    result: dict[int, dict] = {}
    current_year = datetime.now().year

    try:
        driver.get(f"https://statusinvest.com.br/acoes/{ticker.lower()}")
        time.sleep(2)
        _close_popups(driver)

        # Open "Histórico do ativo" panel
        historico_btn = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable(
                (By.XPATH, '//button[@title="Histórico do ativo"]')
            )
        )
        historico_btn.click()
        time.sleep(1)
        _close_popups(driver)

        # Select maximum available date range
        max_btn = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'li[title="Máximo disponível"] a')
            )
        )
        driver.execute_script("arguments[0].click();", max_btn)
        time.sleep(2)
        _close_popups(driver)

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "table-history"))
        )

        # --- Parse column headers to get the list of years ---
        first_table = driver.find_element(By.CSS_SELECTOR, ".table-history")
        header_cells = first_table.find_element(
            By.CSS_SELECTOR, ".tr"
        ).find_elements(By.CSS_SELECTOR, ".th")

        years = [_parse_year(cell.text, current_year) for cell in header_cells]

        # --- Iterate over indicator groups ---
        groups = driver.find_elements(
            By.CSS_SELECTOR,
            ".indicator-historical-container .indicators",
        )

        for group in groups:
            indicator_names = [
                h3.text.strip()
                for h3 in group.find_elements(By.CSS_SELECTOR, "h3.title")
            ]
            table = group.find_element(By.CSS_SELECTOR, ".table-history")
            data_rows = table.find_elements(By.CSS_SELECTOR, ".tr")[1:]  # skip header

            for i, row in enumerate(data_rows):
                if i >= len(indicator_names):
                    break

                col_name = INDICATOR_MAP.get(indicator_names[i])
                if col_name is None:
                    continue  # unknown indicator — skip

                cells = row.find_elements(By.CSS_SELECTOR, ".td")
                for j, cell in enumerate(cells):
                    if j >= len(years) or years[j] is None:
                        continue
                    year = years[j]
                    if year not in result:
                        result[year] = {}
                    result[year][col_name] = _parse_float(cell.text)

        # --- Compute Graham formula for each year ---
        for year, indicators in result.items():
            indicators["graham_formula"] = _calc_graham(
                indicators.get("lpa"), indicators.get("vpa")
            )

        logger.info(f"[{ticker}] Scraped {len(result)} years of indicators")
        return result

    except Exception as exc:
        logger.error(f"[{ticker}] Scraping failed: {exc}")
        return {}

    finally:
        driver.quit()
