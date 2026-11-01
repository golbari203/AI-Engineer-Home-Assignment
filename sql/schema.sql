-- Equity Research Assistant — MySQL schema
-- Every table here exists because of something found in the data. See DESIGN.md.

CREATE DATABASE IF NOT EXISTS era CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE era;

-- ---------------------------------------------------------------------------
-- 1. Reference data
-- ---------------------------------------------------------------------------

-- Fiscal calendars differ: Apple's Q1 FY2026 ended Dec 2025, NVIDIA's Q1 FY2027
-- is Feb-Apr 2026, Costco's year ends near Aug 31, Microsoft's on Jun 30.
-- "Last quarter" is meaningless without this table.
CREATE TABLE IF NOT EXISTS companies (
  ticker            VARCHAR(10)  PRIMARY KEY,
  name              VARCHAR(200) NOT NULL,
  fiscal_year_end   VARCHAR(40),            -- human description
  fy_end_month      TINYINT,                -- 9 = September, etc.
  fy_label_leads    BOOLEAN DEFAULT FALSE,  -- TRUE when FY label runs ahead of calendar (NVDA, AAPL, MSFT)
  aliases           JSON,                   -- ["Apple Inc.", "Apple"]
  is_held           BOOLEAN DEFAULT FALSE,
  is_watchlist      BOOLEAN DEFAULT FALSE
);

-- NVDA 10-for-1 on 2025-06-10 contaminates prices, EPS, quantities and consensus.
-- Any cross-time numeric comparison must pass through this table first.
CREATE TABLE IF NOT EXISTS corporate_actions (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  ticker          VARCHAR(10) NOT NULL,
  action          VARCHAR(40) NOT NULL,     -- 'forward_split'
  effective_date  DATE NOT NULL,
  ratio_num       DECIMAL(12,6) NOT NULL,   -- 10
  ratio_den       DECIMAL(12,6) NOT NULL,   -- 1
  source_document_id INT NULL,
  UNIQUE KEY uq_ca (ticker, action, effective_date)
);

-- ---------------------------------------------------------------------------
-- 2. Documents and their pages
-- ---------------------------------------------------------------------------

-- source_tier implements the adjudication order used when sources conflict:
-- 1 filing > 2 press release > 3 corrected transcript > 4 auto transcript
-- > 5 company presentation > 6 broker note > 7 internal spreadsheet.
CREATE TABLE IF NOT EXISTS documents (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  path            VARCHAR(512) NOT NULL,
  filename        VARCHAR(255) NOT NULL,
  content_hash    CHAR(64) NOT NULL,        -- exact-duplicate detection (MSFT Q2 FY26 appears twice)
  doc_type        VARCHAR(50),              -- filing_10k | filing_10q | press_release | transcript |
                                            -- presentation | broker_note | custodian_statement |
                                            -- corporate_action | internal_workbook
  source_tier     TINYINT NOT NULL DEFAULT 5,
  publisher       VARCHAR(120),             -- 'Westbrook Securities', 'CallScript.io'
  ticker          VARCHAR(10),
  title           VARCHAR(400),
  published_date  DATE,                     -- when the document was issued
  as_of_date      DATE,                     -- what date the content describes
  fiscal_period   VARCHAR(20),              -- 'Q4 FY2026'
  event_key       VARCHAR(120),             -- ticker|event|date — clusters near-duplicates
                                            -- (two vendors' transcripts of the same NVDA call)
  page_count      SMALLINT,
  ocr_page_count  SMALLINT DEFAULT 0,
  language        VARCHAR(10) DEFAULT 'en', -- Banque Leman is fr/en
  warnings        JSON,                     -- ['rotated_pages', 'low_ocr_confidence', ...]
  ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_hash (content_hash),
  KEY ix_ticker_date (ticker, as_of_date),
  KEY ix_event (event_key)
);

-- Page-level storage, because three PDFs have a text layer on some pages and
-- bare images on others (JPM 4Q25 pp.2-3, Apple FY2024 10-K pp.3-8, XOM 10-K pp.7-12).
-- Tracking extraction per page is the only way to notice silent loss.
CREATE TABLE IF NOT EXISTS document_pages (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  document_id     INT NOT NULL,
  page_no         SMALLINT NOT NULL,
  text            MEDIUMTEXT,
  method          VARCHAR(20) NOT NULL,     -- 'text_layer' | 'ocr'
  rotation        SMALLINT DEFAULT 0,       -- XOM 2Q25 scan is 180 degrees
  ocr_confidence  DECIMAL(5,2),
  char_count      INT,
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
  UNIQUE KEY uq_doc_page (document_id, page_no)
);

CREATE TABLE IF NOT EXISTS chunks (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  document_id     INT NOT NULL,
  chunk_index     SMALLINT NOT NULL,
  page_from       SMALLINT,
  page_to         SMALLINT,
  text            MEDIUMTEXT NOT NULL,      -- includes the context header prefix
  embedding       JSON NOT NULL,
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FULLTEXT KEY ft_text (text)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------------
-- 3. Extracted facts  —  conflict detection lives here
-- ---------------------------------------------------------------------------

-- A conflict is just: same (ticker, metric, fiscal_period) with different values.
-- Covers AWS Q4 2025 (23.9 vs 24.7 recast), Costco warehouse counts (913 vs 906),
-- Lilly 2025 revenue (49.3 vs 51.5), NVDA Q1 FY27 guide (41.4bn vs 41.4m),
-- and Microsoft's CFO.
CREATE TABLE IF NOT EXISTS facts (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  ticker          VARCHAR(10),
  metric          VARCHAR(80) NOT NULL,     -- 'revenue', 'eps_diluted', 'cfo_name', 'guidance_revenue'
  fiscal_period   VARCHAR(20),              -- 'Q4 FY2026'
  period_start    DATE,
  period_end      DATE,
  value_num       DECIMAL(22,4),
  value_text      VARCHAR(400),             -- for non-numeric facts (officer names, ratings)
  unit            VARCHAR(20),              -- 'USD_millions', 'USD', 'percent', 'count'
  basis           VARCHAR(30),              -- GAAP | non-GAAP | adjusted | ex_identified_items | managed
  segment         VARCHAR(80),
  is_guidance     BOOLEAN DEFAULT FALSE,
  is_restated     BOOLEAN DEFAULT FALSE,    -- Amazon's recast segment figures
  supersedes_id   INT NULL,
  document_id     INT NOT NULL,
  page_no         SMALLINT,
  stated_on       DATE,                     -- date the source document made the claim
  confidence      DECIMAL(4,3),
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
  KEY ix_conflict (ticker, metric, fiscal_period, basis)
);

-- ---------------------------------------------------------------------------
-- 4. Portfolio
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS accounts (
  id                 INT AUTO_INCREMENT PRIMARY KEY,
  custodian          VARCHAR(120) NOT NULL,
  account_no         VARCHAR(60)  NOT NULL,
  entity             VARCHAR(160) NOT NULL,  -- two different legal entities
  reporting_currency CHAR(3) NOT NULL,       -- USD / CHF
  cash               DECIMAL(22,2),
  total_value        DECIMAL(22,2),
  fx_rate            DECIMAL(16,8),
  as_of_date         DATE,
  UNIQUE KEY uq_account (custodian, account_no)
);

-- source_kind records WHERE a holding came from. The custodian statement and the
-- family office blotter disagree (JPM 7,250 vs 7,520; Costco 900 missing from the
-- books), so both are stored and precedence is applied at query time, never at load.
CREATE TABLE IF NOT EXISTS holdings (
  id                 INT AUTO_INCREMENT PRIMARY KEY,
  account_id         INT NOT NULL,
  ticker             VARCHAR(10) NOT NULL,
  quantity           DECIMAL(20,6) NOT NULL,
  price              DECIMAL(20,6),
  price_currency     CHAR(3),
  market_value       DECIMAL(22,2),
  cost_basis         DECIMAL(22,2),
  value_currency     CHAR(3) NOT NULL,
  fx_rate_used       DECIMAL(16,8),
  as_of_date         DATE NOT NULL,
  source_kind        ENUM('custodian_statement','internal_book') NOT NULL,
  source_document_id INT NOT NULL,
  FOREIGN KEY (source_document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY (account_id) REFERENCES accounts(id),
  KEY ix_pos (ticker, as_of_date, source_kind)
);

CREATE TABLE IF NOT EXISTS transactions (
  id                 INT AUTO_INCREMENT PRIMARY KEY,
  account_id         INT NOT NULL,
  ticker             VARCHAR(10),
  trade_date         DATE,
  settle_date        DATE,
  type               VARCHAR(20) NOT NULL,   -- BUY | SELL | DIV | FEE | TAX | CUSTODY
  quantity           DECIMAL(20,6),
  quantity_is_presplit BOOLEAN DEFAULT FALSE, -- NVDA buy of 1,800 pre-split = 18,000 post
  price              DECIMAL(20,6),
  amount             DECIMAL(22,2),
  currency           CHAR(3),
  fx_rate            DECIMAL(16,8),
  reference          VARCHAR(60),
  memo               VARCHAR(255),
  source_kind        ENUM('custodian_statement','internal_book') NOT NULL,
  source_document_id INT NOT NULL,
  FOREIGN KEY (source_document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY (account_id) REFERENCES accounts(id),
  KEY ix_tx (ticker, trade_date)
);

-- Prices.xlsx is explicitly NOT split-adjusted (NVDA 975.90 -> 84.55 between
-- month-ends). The flag stops anyone computing a 91% crash.
CREATE TABLE IF NOT EXISTS prices (
  ticker          VARCHAR(10) NOT NULL,
  price_date      DATE NOT NULL,
  close_price     DECIMAL(20,6) NOT NULL,
  currency        CHAR(3) DEFAULT 'USD',
  split_adjusted  BOOLEAN NOT NULL DEFAULT FALSE,
  source_document_id INT,
  PRIMARY KEY (ticker, price_date, split_adjusted)
);

-- FX_rates.xlsx offers both directions in different sheets with different number
-- formats. Store one canonical direction and derive the other.
CREATE TABLE IF NOT EXISTS fx_rates (
  rate_date   DATE NOT NULL,
  base_ccy    CHAR(3) NOT NULL,   -- USD
  quote_ccy   CHAR(3) NOT NULL,   -- CHF
  rate        DECIMAL(16,8) NOT NULL, -- quote units per 1 base
  source      VARCHAR(60),
  PRIMARY KEY (rate_date, base_ccy, quote_ccy)
);

-- ---------------------------------------------------------------------------
-- 5. Street data
-- ---------------------------------------------------------------------------

-- eps_basis differs per company (GAAP, non-GAAP, Adjusted, ex identified items).
-- Comparing an estimate to an actual on the wrong basis is the classic error.
CREATE TABLE IF NOT EXISTS consensus (
  id                INT AUTO_INCREMENT PRIMARY KEY,
  ticker            VARCHAR(10) NOT NULL,
  fiscal_period     VARCHAR(20) NOT NULL,
  period_end        DATE,
  revenue_consensus DECIMAL(20,4),
  eps_consensus     DECIMAL(12,4),
  eps_basis         VARCHAR(30),
  n_estimates       SMALLINT,
  report_date       DATE,
  revenue_actual    DECIMAL(20,4),
  eps_actual        DECIMAL(12,4),
  as_of_date        DATE NOT NULL,   -- the workbook is as of 2026-06-30
  source_document_id INT,
  UNIQUE KEY uq_cons (ticker, fiscal_period, as_of_date)
);

-- The vendor's aggregate across all analysts: one target and the buy/hold/sell
-- counts per ticker. Not research on file, and never mixed with research_ratings.
CREATE TABLE IF NOT EXISTS street_targets (
  ticker          VARCHAR(10) NOT NULL,
  consensus_pt    DECIMAL(12,2),
  buy             SMALLINT,
  hold            SMALLINT,
  sell            SMALLINT,
  last_price      DECIMAL(12,4),
  as_of_date      DATE NOT NULL,
  PRIMARY KEY (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS research_ratings (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  ticker          VARCHAR(10) NOT NULL,
  house           VARCHAR(80) NOT NULL,
  analyst         VARCHAR(80),
  rating          VARCHAR(30),      -- house-specific: Buy / Overweight / Market Perform
  rating_scale    VARCHAR(60),      -- the house's 3-point scale, not comparable across houses
  price_target    DECIMAL(12,2),
  price_at_note   DECIMAL(12,2),
  horizon         VARCHAR(30),
  note_date       DATE NOT NULL,
  document_id     INT,
  KEY ix_rating (ticker, note_date)
);

-- ---------------------------------------------------------------------------
-- 6. Data quality
-- ---------------------------------------------------------------------------

-- Populated by reconcile.py. Answers "is anything wrong with my records?" and
-- gives the agent an honest basis for caveats.
CREATE TABLE IF NOT EXISTS reconciliation_exceptions (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  check_name    VARCHAR(80) NOT NULL,
  severity      ENUM('info','warning','error') NOT NULL,
  ticker        VARCHAR(10),
  account_id    INT,
  description   TEXT NOT NULL,
  source_a      VARCHAR(255),
  source_b      VARCHAR(255),
  value_a       DECIMAL(22,4),
  value_b       DECIMAL(22,4),
  delta         DECIMAL(22,4),
  detected_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_log (
  id             INT AUTO_INCREMENT PRIMARY KEY,
  question       TEXT,
  tool_calls     JSON,
  input_tokens   INT,
  output_tokens  INT,
  latency_ms     INT,
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
