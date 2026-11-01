# Equity Research Assistant

An assistant for a family office. The investor asks questions in plain English,
such as "How much Costco do I own?" or "Who is Microsoft's CFO?", and gets
answers drawn only from the documents in the `portfolio/` folder. Every figure
shows where it came from, and one click opens the page it was taken from.

## Before you start

You need three things:

1. **Docker Desktop**, installed and running. On a Mac: `brew install --cask docker`,
   then open the Docker app once and wait until the whale icon in the menu bar
   stops moving.
2. **An Anthropic API key** with credit on the account.
3. **The `portfolio/` folder** from the assignment: 58 PDFs and 4 spreadsheets.

That's all. Python, the database, the OCR tools and the search model all run
inside Docker, so nothing else needs installing.

## Run it

```bash
# 1. Unzip the project and go into it
unzip equity-research-assistant.zip
cd equity-research-assistant

# 2. Give it your API key, in a .env file or in your shell
cp .env.example .env               # then edit .env: AIEHA_API_KEY=sk-ant-...
# or: export AIEHA_API_KEY=sk-ant-...

# 3. Copy in the documents
cp -r /path/to/portfolio ./portfolio

# 4. Build and start
docker compose up --build
```

When the log says `You can now view your Streamlit app`, open
**<http://localhost:8501>**.

### What to expect the first time

The first start takes **10 to 15 minutes**, and only the first one does. In that time the app:

- builds its containers;
- reads all 62 documents, including six scans that need OCR;
- asks the AI model to pull the key figures out of each document. This costs a
  few dollars, once.

Everything it learns is saved, so after that **a restart takes about 15
seconds**. The first page load after a restart takes another 20 seconds or so
while the search index warms up.

### Everyday commands

```bash
docker compose up -d               # start in the background
docker compose logs -f app         # watch the logs (Ctrl+C stops watching, not the app)
docker compose down                # stop; your data is kept
docker compose down -v             # stop and erase everything, for a completely fresh start
```

If you change the code, start it with `docker compose up -d --build`.

## Using the app

There are four pages:

- **Ask** is the chat. Each answer lists its sources; click **Inspect** to read
  the exact page a figure came from. If the documents don't cover a question,
  the assistant says so instead of guessing.
- **Portfolio** shows what you own and where: totals, the two accounts, how
  concentrated the holdings are, and any places where your own records
  disagree with the custodians.
- **Companies** has a profile for every company you own or have research on:
  your position, the figures from its reports, analyst estimates, broker
  notes and its documents.
- **Sources** lists every document. Search them, read them, or upload new ones.

**Adding new documents.** Drop a PDF into the `portfolio/` folder, or upload it
on the Sources page. Within seconds the assistant can answer questions about it,
with no restart. That works for companies it hasn't seen before too: they get a
profile of their own. If you replace one of the spreadsheets, say a corrected
`Holdings and trades.xlsx`, the record checks re-run automatically.

### Good questions to start with

| Ask | Why it's interesting |
|---|---|
| How much Costco do I own? | It's split across two accounts and two currencies. The easy wrong answer is 1,400 shares. |
| How many NVIDIA shares do I hold? | Without the 2025 stock split, the trade history adds up to minus 2,200 shares. |
| Is anything wrong with my records? | It finds seven issues by arithmetic, not guesswork. |
| Who is Microsoft's CFO? | Some later documents still name the CFO who left. |
| What was Lilly's 2025 revenue? | A broker and the company give different numbers. |
| What's Meta's outlook? | There's nothing on file about Meta, so it says so. |

## Checking that it works

This runs the automated tests and the evaluation inside Docker. It doesn't call
the AI model, so it costs nothing. Run it from the project folder:

```bash
docker compose run --rm -v "$PWD/portfolio:/mnt/user-data/uploads:ro" \
  -e ERA_OFFLINE=1 -e HF_HUB_OFFLINE=1 app sh -c '
    for t in tests/test_*.py; do python3 "$t"; done   # 201 tests
    python3 eval/run_eval.py                           # 42 of 42 checks
    python3 eval/run_eval.py --mysql                   # the same 42, read from the database
    python3 eval/cost.py                               # what a question costs'
```

To grade the assistant's real answers as well, add `--live` to the eval. That
one calls the model, so it needs credit.

## Settings

Put these in `.env` or your shell. Only the API key is required.

| Setting | Default | What it does |
|---|---|---|
| `AIEHA_API_KEY` | — | Your Anthropic API key |
| `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | `era` / `era` / `era` | Database login |
| `MYSQL_PORT` | `3306` | The port the database uses on your machine. Change it if 3306 is taken |
| `ERA_OFFLINE` | off | `1` swaps the search model for a simple word-matcher (for tests only) |
| `HF_HUB_OFFLINE` | off | `1` stops the search model checking online for updates |

**Looking inside the database:** run
`docker compose exec mysql mysql -uera -pera era`, or connect any SQL tool to
`localhost:3306` with user and password `era`. The data lives in a Docker
volume called `equity-research-assistant_mysql_data`.

## If something goes wrong

| What you see | What to do |
|---|---|
| `docker: command not found` or `failed to connect to the docker API` | Docker isn't installed or isn't running. Open Docker Desktop and wait for it to start. |
| `required variable AIEHA_API_KEY is missing` | Add the key to `.env`, or `export` it in the same terminal. |
| The chat says the AI service has run out of credit | Add credit in the Anthropic console. Everything except the chat keeps working. |
| `port is already allocated` for 3306 | Something else is using that port. Set `MYSQL_PORT=3307` in `.env`. |
| `Could not reach the Hub` warnings at startup | Harmless. Set `HF_HUB_OFFLINE=1` to hide them. |
| A document has no figures after an API error | Restart the app; it retries anything that failed. |

## Running without Docker

You can, but Docker is easier: one command, and it doesn't depend on what's
already on the machine. To run directly on a Mac (on Linux, use
`apt-get install poppler-utils tesseract-ocr python3.12 python3.12-venv mysql-server`
for step 1):

```bash
cd equity-research-assistant

# 1. Install the tools: PDF reading, OCR, Python 3.12 and MySQL
brew install poppler tesseract python@3.12 mysql

# 2. Start MySQL and create the database the app expects
brew services start mysql
mysql -u root -e "CREATE DATABASE IF NOT EXISTS era;
  CREATE USER IF NOT EXISTS 'era'@'localhost' IDENTIFIED BY 'era';
  GRANT ALL ON era.* TO 'era'@'localhost';"

# 3. Install the Python packages in a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Settings
export AIEHA_API_KEY=sk-ant-...
export PORTFOLIO_DIR="$PWD/portfolio" MYSQL_HOST=127.0.0.1 PYTHONPATH="$PWD"

# 5. Load the documents, then start the app
python scripts/bootstrap.py --folder ./portfolio
streamlit run app/ui.py
```

Then open <http://localhost:8501>. In a new terminal, run
`source .venv/bin/activate` and the two `export` lines again before step 5.

Some differences from Docker:

- **Everything lives on your machine.** That's the packages, the database and
  the saved work (in `~/.era_cache`).
- **It downloads the search model on first use**, about 130 MB.
- **The first run starts from zero.** It reads, OCRs and extracts figures again,
  about 10 minutes, and the figure step needs API credit. Without credit the
  app still runs, just without the extracted figures.
- **Port 3306 must be free.** If the Docker version is running, stop it first
  with `docker compose down`.
- **The root password:** the `mysql -u root` line assumes no root password,
  which is Homebrew's default. Add `-p` if yours has one.
- **For the full test run, use Docker.** Without it, the tests that read the
  real documents skip themselves.

## What's where

```
app/
  ui.py, ui_data.py          the four pages, and the data behind them
  agent.py, tools.py         the assistant and the seven lookups it can use
  retrieval.py               document search, by meaning and by keyword
  reconcile.py               the record checks (your files vs the custodians)
  pipeline.py, watcher.py    reading a new file; watching the folder for them
  db.py, offline_store.py    the database, and the same data loaded straight from files
  filenames.py               matching file names however they're spaced or capitalised
  ingest/                    reading PDFs and spreadsheets, OCR, classification, figures
sql/schema.sql               the database tables
scripts/bootstrap.py         what runs at start-up: load the documents, extract the figures
eval/                        the 42 checks and the cost estimate
tests/                       201 tests; no API key needed
```

[DESIGN.md](DESIGN.md) explains what was built and why.
