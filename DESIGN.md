# Design

## What I built

A chat assistant for an investor who isn't technical. It answers only from the
62 files in `portfolio/`, and it shows its working: every figure comes with its
source, and one click opens the page it was taken from.

It's a Streamlit app on a Python backend with MySQL, and one command starts it.
Alongside the chat there are three pages that show what the assistant is
reading: the portfolio, a profile for each company, and the documents
themselves. The investor can check an answer without having to ask a second
question. New files can be dropped in while the app is running; they're read,
understood and searchable within seconds, even when they're about a company it
hasn't seen before.

## What I learned about the data

I read all 62 files before writing any code. They're messy in the ways real
files are, and each problem below changed the design.

- **The portfolio's own records disagree.** The custodian says the family owns
  7,250 JPMorgan shares; the family office's spreadsheet says 7,520. The
  Trust's 900 Costco shares are on the Swiss bank statement but missing from
  the spreadsheet. The neatest file is the least reliable one, so I keep every
  source separately and never merge them.
- **Some PDFs hide their best pages.** Three files mix normal text pages with
  scanned images. Standard text extraction skips the images without any error,
  and those pages hold JPMorgan's income statement, Apple's management
  discussion and Exxon's financial statements. One scan is upside down. Two
  have file names that say nothing. One file is an exact copy of another.
- **Documents contradict each other.** NVIDIA's automatic call transcript says
  guidance of "$41.4 million"; the corrected transcript says billion.
  Microsoft announced a new CFO three times, yet three later documents still
  name the old one. A broker and Lilly disagree on Lilly's 2025 revenue.
  Amazon restated its segment figures halfway through the year. You can't
  average these. Someone has to decide which source to believe, and say why.
- **The numbers carry traps.** NVIDIA's 10-for-1 stock split breaks prices,
  earnings per share and share counts: without adjusting for it, the trade
  history adds up to minus 2,200 shares. Companies report earnings on
  different bases (GAAP, non-GAAP, adjusted). The spreadsheets mix date
  formats, European decimal commas and exchange rates quoted both ways round.
- **Running it on a second machine taught me two more things.** The same files
  arrived with spaces in their names instead of underscores, which broke the
  spreadsheet loader, six tests and half the evaluation. File names are now
  matched however they're spaced or capitalised. And companies often print a
  rounded figure in the headline and the exact one in a table ("$57.0 billion"
  vs 56,962), so a simple comparison mistakes agreement for a dispute.

## Why it's built this way

**Different questions need different tools.** "How much Costco do I own?" has
one right answer, and it must be complete. Searching documents for it is the
wrong approach: forty PDFs mention Costco, and a search-only assistant
confidently answers 1,400 shares, with citations. So the data lives in three
places, each suited to one kind of question:

- **Portfolio tables** for what the family owns. Every row remembers which
  source it came from.
- **A figures table** for numbers stated in the documents, each with its
  period and accounting basis. "Do the documents disagree?" then becomes a
  simple lookup. It holds 1,354 figures from 54 documents, and it catches
  every known dispute.
- **Searchable text** for questions that don't have a single right value, like
  "why did margins fall?"

**The assistant uses seven fixed lookups instead of writing its own database
queries.** A correct answer here depends on details that are easy to get
wrong: positions appear twice (custodian and family office), NVIDIA's shares
need the split adjustment, and earnings must be compared on the same basis.
The lookups build those rules in once, so the model can't forget them.

**Simple rules first, the AI model second, and never twice for the same
file.** Much of what matters is printed plainly on each document: "CORRECTED
TRANSCRIPT", "FORM 10-Q", the date. Reading those with simple rules is free and
gives the same answer every time. The model only fills the gaps, like naming a
company it hasn't seen, and pulls out the figures. Its answers are saved
against each file's contents, so every document costs one model call, ever.

**When documents disagree, there's a fixed order of trust:** official filings,
then company press releases, corrected transcripts, automatic transcripts,
presentations, and broker notes last. The model never decides a document's
place in that order.

**Search works two ways at once.** A small search model, running inside the
app at no cost, finds passages by meaning. Keyword search catches exact names
the model wouldn't know, like a fictional 2026 drug-pricing law. The two lists
of results are merged into one.

**The design is built for trust, not decoration.** The investor can't judge
how the system works, only whether its answers hold up. So sources open to the
exact page, record problems stay visible on screen, analysts' ratings are
shown side by side and never averaged, and a question the documents can't
answer is clearly labelled. Streamlit kept the whole project in Python, which
left the time for getting the numbers right.

## How I know it works

- **201 automated tests**, which run without an API key because the AI model
  is replaced by a stand-in. They check the rules rather than the model: the
  model can't overrule the simple rules, no file is sent to the model twice,
  and a figure with an invalid accounting basis is rejected.
- **42 scenario checks**, each built from a specific trap in the data:
  portfolio maths, conflicting sources, timing, unanswerable questions,
  fictional versus real facts, scanned pages and accounting bases. All 42 pass,
  whether the data comes from the files or from the database. The checks can
  tell good from bad: when the search may return only its single best match
  instead of several, the score drops to 83%. And when I deliberately broke the
  stock-split adjustment, they failed with exactly the minus 2,200 shares from
  the data study.
- **The record checks find seven real problems** in the data using arithmetic
  alone. Because those results are known, they also catch regressions for free.
- **A 90-point checklist run against the live app**, including 25 real
  questions. Every portfolio figure and every fictional-versus-real answer was
  right, and the assistant invented nothing when the files had no answer. I
  also timed and tested the things a demo depends on:
  - a cold start takes 290 seconds, plus about 6 minutes of one-off figure
    extraction;
  - a PDF dropped into the folder can be searched about a second after it
    finishes copying;
  - correcting JPMorgan in the spreadsheet clears the warning in about 3 seconds;
  - a completely wiped database rebuilds identically.

Testing also found real bugs, and fixing them improved the design. The app was
searching with a stand-in model meant only for tests. The database recorded
the Swiss account in dollars instead of francs, which understated the
portfolio by $645k, and it dropped the analysts' price targets. And on a Mac,
a file still being copied can look finished, so the app now checks that a PDF
is actually complete before reading it.

**What's not proven yet.** Before the figures table was filled, four answers
fell short: one didn't mention NVIDIA's second transcript, one didn't mention
the broker's Lilly figure, one quoted Exxon ratings from a data vendor, and one
didn't explain a timing gap for UnitedHealth. I haven't re-tested them since,
because the API credit ran out. A document added while the app is running is
searchable straight away but only reaches the database at the next restart.
And the two bank statements are loaded from a hand-checked copy rather than
read live.

## What it costs to run

From real prompts and lookups, measured by `eval/cost.py`:

| | |
|---|---|
| A typical question (three model calls) | **about 2.5 cents, or 1.7 cents with prompt caching** |
| 400 questions a month | about $7 |
| Reading the documents | once per document: a few dollars for the whole set, then free on every restart |
| Search, OCR and database | run on the machine, so no ongoing cost |

The biggest savings, in order: caching the instructions the model sees on every
call (about 1,500 tokens), a cheaper model for classifying documents and
extracting figures, and sending fewer search results with each question.

## What I'll do next

1. **Re-run the live checklist** now that the figures are in, and fix the four
   answers that fell short.
2. **Treat rounding as agreement**, so "$57.0 billion" and 56,962 aren't flagged
   as a dispute, and "disputed" means the documents really disagree.
3. **Warn about contradictions as they arrive.** When a new document disagrees
   with one already on file, tell the investor straight away rather than wait
   for them to ask.
4. **Close the database gaps:** save new documents to the database
   immediately, read the bank statements live, and start the folder watcher
   with the app rather than on the first visit.
5. **Production basics:** a proper register of companies and share classes,
   better ranking of search results, live prices, and separate access for the
   two legal entities, with a log of who asked what.

**What I chose not to build:** a custom web frontend, a separate search
database, a fine-tuned model, several cooperating agents, and streamed answers.
With two to three days, I spent the time on making sure the numbers are right.
