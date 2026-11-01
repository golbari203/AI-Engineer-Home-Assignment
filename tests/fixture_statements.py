"""Custodian statement data in the shape statements.py will produce.

Used to exercise reconcile.py before the LLM-backed statement extractor is
wired up. Values transcribed from the two statements in the portfolio folder.
"""

NORTHGATE = {
    "custodian": "Northgate Securities LLC",
    "account_no": "4471-8820",
    "entity": "Harbor Oak Holdings LLC",
    "as_of_date": "2026-06-30",
    "currency": "USD",
    "price_currency": "USD",
    "fx_rate": None,
    "cash": 4_022_308.25,
    "total_value": 17_194_255.75,
    "positions": [
        {"ticker": "AAPL", "quantity": 12_000, "price": 209.07,
         "market_value": 2_508_840.00, "cost_basis": 1_958_160.00},
        {"ticker": "MSFT", "quantity": 6_500, "price": 400.51,
         "market_value": 2_603_315.00, "cost_basis": 1_941_095.00},
        {"ticker": "NVDA", "quantity": 14_000, "price": 141.15,
         "market_value": 1_976_100.00, "cost_basis": 1_091_580.00},
        {"ticker": "AMZN", "quantity": 12_000, "price": 170.90,
         "market_value": 2_050_800.00, "cost_basis": 1_664_970.00},
        {"ticker": "JPM", "quantity": 7_250, "price": 231.89,
         "market_value": 1_681_202.50, "cost_basis": 1_442_460.00},
        {"ticker": "XOM", "quantity": 11_000, "price": 96.75,
         "market_value": 1_064_250.00, "cost_basis": 1_151_480.00},
        {"ticker": "COST", "quantity": 1_400, "price": 919.60,
         "market_value": 1_287_440.00, "cost_basis": 1_114_946.00},
    ],
}

LEMAN = {
    "custodian": "Banque Léman & Cie SA",
    "account_no": "CH-5521.03",
    "entity": "Harbor Oak 2019 Family Trust",
    "as_of_date": "2026-06-30",
    "currency": "CHF",
    "price_currency": "USD",
    "fx_rate": 0.7801,
    "cash": 1_126_737.44,
    "total_value": 2_286_806.99,
    "positions": [
        {"ticker": "UNH", "quantity": 2_600, "price": 253.63,
         "market_value": 514_427.58, "cost_basis": 736_029.24},
        {"ticker": "COST", "quantity": 900, "price": 919.60,
         "market_value": 645_641.96, "cost_basis": 679_589.74},
    ],
}

ALL = [NORTHGATE, LEMAN]
