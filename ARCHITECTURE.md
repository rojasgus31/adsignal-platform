# AdSignal — architecture decision record

This is the reasoning behind the choices, not a tour of the code. Every
number below is reproducible from this repository; where a claim rests on a
vendor's published limits rather than on something measured here, it says so.

---

## 1. Two stores: DuckDB for transformation, Postgres for serving

The pipeline transforms with dbt against DuckDB
(`src/adsignal/warehouse/duckdb_io.py` is the only module that opens a DuckDB
connection) and loads only the aggregated daily marts into Postgres
(`src/adsignal/cli/load_marts.py`, `src/adsignal/warehouse/loader.py`).

The two workloads have nothing in common. The transformation step scans every
generated event — ten million by default (`ADSIGNAL_TARGET_EVENTS`) — column
by column, once a night, with no latency requirement at all. The serving step
answers point reads for a handful of rows, on demand, with a human waiting.
An embedded columnar engine reading Parquet is close to ideal for the first
and useless for the second (nothing to connect to, no concurrency story). A
hosted row store is right for the second and would make the first an
expensive, slow, connection-bound ETL.

The second reason is operational: the pipeline runs on an ephemeral runner,
so the DuckDB file it builds disappears with the job. Postgres is where the
result outlives the process that computed it. That is precisely what
`cli/load_marts.py` exists for.

The cost is a copy step and the chance of the two drifting. The mitigation is
that the load is a full refresh inside one transaction — truncate and
reinsert — so the serving table is either the previous night's marts or this
night's, never a blend.

---

## 2. Precomputed, not served live

Every endpoint except `/search`, `/score` and `/ask` returns rows the
pipeline already computed and stored. Visitor traffic against `/campaigns`,
`/anomalies`, `/models` and `/benchmark` triggers no inference, no LLM call
and no external request, at any load.

`/score` is the deliberate exception to the rule this section states, and it
earns its place by answering the objection the rule invites: a served row
proves storage, not a working model. It runs the fitted detector and triage
estimators in the API process against features built for the requested
campaign-day, and it is the tightest-throttled endpoint in the API for
exactly that reason.

This is a deliberate inversion of the obvious design. The expensive work —
generation, dbt build, training, scoring, drift, embeddings, the narrative —
happens once a night on a runner that nobody is waiting on, where a slow or
failed step costs a retry rather than a timeout in someone's browser. What
the API does is read.

It also bounds the blast radius of a bad actor. The only three endpoints
that do work per request are the three that are rate-limited
(`src/adsignal/api/routes/search.py`, `src/adsignal/api/routes/score.py`,
`src/adsignal/api/routes/ask.py`), and `/ask` is additionally cached. Everything else is a bounded, paginated read
against an indexed table.

---

## 3. CPU-only models, and why TurboQuant was rejected

Both supervised models are scikit-learn (`IsolationForest`,
`HistGradientBoostingClassifier` — `src/adsignal/ml/`). The embedder is
`BAAI/bge-small-en-v1.5` quantised to int8 ONNX via fastembed: roughly 90 MB
of weights, no PyTorch, no CUDA (`src/adsignal/ml/embeddings.py`). The whole
system is designed to run inside a 512 MB container on free CPU hosting —
see §8 — and a GPU dependency would end that immediately.

TurboQuant was considered and rejected. It is KV-cache quantization shipped
as Triton kernels for vLLM: it needs NVIDIA hardware, and what it optimizes
is long-context *generative* inference — the memory growth of a KV cache
across thousands of decoded tokens. This workload has no KV cache. `/search`
runs a single forward pass of a 384-dimension bi-encoder over one short
query string; the nightly embedding asset runs the same pass in a batch.
There is no generation, no context to cache, and no NVIDIA device. Adopting
it would have added a hardware floor in exchange for an optimization that
does not apply.

---

## 4. The release gate is ROC AUC, not recall

The gate that is actually enforced is in `src/adsignal/ml/gate.py`
(`gate_failures`), and it checks exactly two floors, both declared in
`src/adsignal/ml/evaluation.py`:

- triage macro F1 ≥ `MINIMUM_MACRO_F1` (0.45)
- detector ROC AUC ≥ `MINIMUM_DETECTOR_AUC` (0.65)

Recall is reported (`DetectorMetrics.recall`, printed by `_detector_lines`)
but nothing gates on it, because recall on this detector is base-rate
dependent by construction. `IsolationForest`'s contamination — and therefore
its flagging threshold — is derived from the training window's own observed
anomaly rate (`AnomalyDetector._resolve_contamination`). The comment on
`MINIMUM_DETECTOR_AUC` records the consequence: **the same detector measured
0.60 recall on a mart with a ~29% anomaly rate and 0.30 on one with ~5%**. A
fixed recall floor under that design is not a regression gate; it is a number
that moves whenever the dataset's anomaly rate moves, with no change to the
model at all.

A contamination sweep on real pipeline data confirmed this is not a tuning
problem either: F1 is maximised at the derived contamination, and reaching
recall 0.5 from there **costs more than half the detector's precision**.
(That is the figure `evaluation.py` commits to. A more precise
precision-at-0.5-recall number circulated during review; it is not recorded
anywhere in this repository, so it is not asserted here — re-run the sweep if
you need it exactly.)

AUC asks the threshold-independent question instead: does the detector's
continuous score still rank real anomalies above normal days at all? That is
what a regression gate should ask. The 0.65 floor sits below the ~0.76 AUC
measured on real pipeline data and well above 0.5.

The detector is gated separately from the classifier because it is a hard
pre-filter — the scoring asset only triages rows the detector flagged. A
detector whose signal has degraded starves the classifier of real anomalies
while triage macro F1 still looks healthy, so triage F1 alone cannot catch
that failure. Both floors run in `.github/workflows/eval.yml` via
`python -m adsignal.cli.quality_gate`.

---

## 5. `/ask` generates SQL, and three controls contain it

`/ask` used to match a question against four named query templates. That
design could not be injected into by construction — its selector emitted one
of four fixed strings and never a query — and it could not answer anything
its author had not thought of. The project owner asked for the second thing,
so the first was given up deliberately.

The endpoint now sends the question and a description of seven tables to
Gemini and runs the SQL that comes back. **The structural guarantee is gone.
Three independent controls replace it, and each is sufficient on its own.**

1. **A read-only Postgres role.** `warehouse/ask_reads.ensure_readonly_role`
   creates `adsignal_ro`: `LOGIN`, `SELECT` on the seven allowlisted tables,
   ownership of nothing, and an explicit `REVOKE ALL` on `model_artifacts`
   and `entity_embeddings`. Generated SQL runs only on
   `ADSIGNAL_POSTGRES_READONLY_DSN`; the admin DSN is never passed to the
   executor. `tests/test_ask_reads.py` opens a connection with this role and
   issues `DROP TABLE dim_campaign` in an ordinary read-write transaction —
   no validator, no read-only transaction — and asserts Postgres refuses it.

   **The grants are applied by a deployment command, not by the API.**
   `python -m adsignal.cli.bootstrap_readonly_role` reads the admin DSN and
   the read-only DSN from the environment, takes the role name and password
   out of the latter, applies the grants and then runs the same privilege
   probe `/ask` makes before its first query. It is a command rather than a
   request-time action because applying grants needs the admin DSN, and the
   serving process holding one would delete the point of the role. It is
   idempotent and belongs in the deploy hook.

   Nothing in `src/` called `ensure_readonly_role` until that command
   existed; only `tests/conftest.py` did. Every test of the role passed
   while a real deployment refused every question with
   `rule: "database_permission"`, because the fixture was doing the
   deployment's job. `tests/test_deploy_bootstrap.py` is the test that
   closes that: it strips the role's grants, asks a question and asserts the
   refusal, runs only the documented command, and asserts the same question
   is answered. `tests/test_wiring.py` is the general guard — a written-down
   list of functions that shipped code must call, checked against every call
   expression under `src/`.
2. **A read-only transaction with a 5 s statement timeout, pinned parsing
   settings, and the extended query protocol.** Every execution sets `SET LOCAL transaction_read_only =
   on` and `SET LOCAL statement_timeout = '5s'`, so a write slips past
   nothing and a runaway cross join dies on its own. The query is sent with
   `prepare=True`, which forces psycopg onto the extended protocol: a
   parameterless statement would otherwise go over the *simple* protocol,
   where Postgres itself accepts several statements separated by semicolons.
   On the extended protocol the server refuses multiple commands outright, so
   stacking is impossible at the wire level rather than only by rule — and
   Postgres parses the statement with its own grammar, which is the grammar
   that ultimately matters.

   Three settings are pinned for the duration of every query, because the
   validator's correctness silently depends on them.
   `standard_conforming_strings = on` is the one that matters: with it off,
   Postgres reads a backslash in an ordinary literal as an escape, so
   `SELECT '\' AS a, ', 1 AS z FROM model_artifacts --' AS c FROM
   dim_campaign` — approved as reading `dim_campaign` — reads
   `model_artifacts` instead. sqlglot's Postgres dialect always assumes `on`,
   so the round-trip check *structurally cannot* see this: both parses agree
   with each other and disagree with the server. The default is `on`, but
   `postgresql.conf` or `ALTER ROLE adsignal_ro SET …` can change it where
   nobody is looking. `backslash_quote = off` refuses `\'` inside `E''`
   strings, the exact construction behind the escape-string regression.
   `search_path = public` makes the table allowlist mean what it says, since
   the validator refuses qualified names and would otherwise leave name
   resolution to the role's own configuration.

   `search_path` is `pg_catalog, public, pg_temp`, and the last entry is the
   one that does the work. Postgres searches the temp schema **first** unless
   the path names it, and `adsignal_ro` can create temp tables since
   TEMPORARY is granted to PUBLIC by default — so a temp `dim_campaign` would
   shadow the real one. Naming `pg_catalog` alone does nothing: measured
   against the server with a temp `dim_campaign` present, `pg_catalog,
   public` resolves the name to `pg_temp_3` exactly as bare `public` does,
   and only listing `pg_temp` last resolves it to `public`. Nothing reaches
   this through `/ask` (a read-only transaction refuses `CREATE TEMP TABLE`,
   and each request is its own transaction), but a pin that does not do what
   its comment says is worth less than no pin — and the first version of this
   one did not, with a test that passed vacuously because the pooled
   connection had no temp schema. A deployment that wants the hole closed at
   the source should `REVOKE TEMPORARY ON DATABASE <db> FROM PUBLIC`; this
   project does not do that to someone else's database.

   `DateStyle` and `TimeZone` are pinned too, for a different reason: they
   change the *answer* rather than the meaning. `'01/02/2024'::date` is a
   different day under `DMY`, and `TimeZone` moves `CURRENT_DATE`, `now()`
   and every `timestamptz` comparison, so the same approved query returns
   different rows on a differently configured server. `ISO, MDY` is
   Postgres's own default, so pinning it changes nothing on a default server.

   Settings deliberately **not** pinned: `bytea_output` and the `lc_*` family
   change how a value is *rendered*, not which rows come back, and psycopg
   returns typed Python objects that `_jsonable` converts itself.

   **The row cap is not a size cap.** An aggregate builds one value from
   every row — `query_to_xml` returned 8 MB in a single column, and
   `jsonb_agg`, `array_agg` and `string_agg` can all do it legitimately — so
   `MAX_RESULT_BYTES` bounds the whole answer and refuses rather than
   truncating. A silently shortened answer is a wrong answer that looks
   right, which is the one thing showing the user the SQL is meant to
   prevent. It counts **encoded UTF-8 bytes**: the first version summed
   `len(str(value))`, which is characters, so `repeat('€', 900000)` passed a
   cap named for bytes and served 2.7 MB. It runs after the rows are
   materialised, so it bounds what is *served*, not what the API process
   allocates — a 60 MB value is built and then refused. Bounding allocation
   would mean streaming and measuring as rows arrive, which an endpoint
   capped at 200 rows does not justify; the statement timeout is what bounds
   the work on the server.
3. **A pure validator** (`warehouse/sql_guard.py`, no I/O). One statement
   only, opening with `SELECT` or `WITH`; no comments; no writing,
   administrative or catalog construct; no table outside the allowlist; a
   `LIMIT` of 200 on the outermost query, refused when larger.

   **It parses with sqlglot rather than matching patterns, because patterns
   were not enough.** The first implementation was a regex scanner, and
   review found three inputs that reached `model_artifacts` through it:
   `FROM "model_artifacts"` (a quoted identifier never matched the FROM
   pattern), `FROM dim_campaign, model_artifacts` (only the first item
   follows the word FROM), and a string literal containing
   `, model_artifacts as (` that registered the name as a trusted CTE. Each
   was a place where a pattern encoded a guess about SQL's grammar and the
   guess was wrong; a fourth pattern would have been a fourth guess.
   **What runs is what the validator understood** — the SQL is rendered back
   from the parsed tree, so anything sqlglot did not model is absent from
   what reaches Postgres rather than passing through unexamined. `LIMIT ALL`
   is the concrete case: the scanner saw the word LIMIT and appended no cap.

   **And the rendering is itself checked, because rendering was a hole.**
   That reasoning holds in one direction only: sqlglot can *emit* something
   it models differently from Postgres. Review found
   `SELECT E'\' AS a, ', b FROM model_artifacts --' AS c FROM dim_campaign`
   accepted, because sqlglot rendered the escape string as `e'\'`, where
   Postgres reads `\'` as an escaped quote rather than a terminator — so the
   literal's boundary moved between what was validated and what would have
   run, and the rendered query re-parses to `FROM model_artifacts` with the
   cap commented out. Every query is now validated **twice**, once as written
   and once as rendered, and refused unless the two agree on their real table
   set, their comment absence and their effective cap, with the rendering
   required to be a fixed point.

   **Table references are resolved per scope.** A `WITH` nested inside a
   sub-select declares a name only for that sub-select; collecting CTE names
   tree-wide let one launder the same name for the outer query. sqlglot's
   scope machinery decides which references are real tables, and a reference
   the analyser cannot classify is treated as a real table rather than
   waved through.

   **Both spellings of the row bound count.** `FETCH FIRST n ROWS ONLY` is
   LIMIT in standard clothing; knowing only `Limit` turned a rule that had
   been failing closed into one that failed open. A bound that is not a plain
   number of rows — `FETCH FIRST 50 PERCENT ROWS ONLY` — is refused as a
   `row_limit` rather than left to fail as a parse error at the server.

   **Functions are an allow-list, and that shape is the lesson.** The rule
   was a deny-list — `pg_*`, `information_schema`, `dblink` — and review
   walked past it with `query_to_xml('SELECT * FROM model_artifacts', …)`,
   which matches none of those patterns, takes SQL as a *string argument*,
   and returns a whole table as a single value. The approved table set said
   `dim_campaign`; Postgres executed a second query nobody had read, and the
   result was one 8 MB value, so the row cap and the bounded fetch fell along
   with the allowlist. `table_to_xml`, `schema_to_xml`, `database_to_xml` and
   `query_to_xmlschema` are the same hazard in four more spellings, and the
   next extension installed adds more — a deny-list must name a thing before
   it can refuse it, and that set is not knowable.

   sqlglot parses a function it models into a typed node and everything else
   into `Anonymous`, so the rule is now: **if the parser does not recognise
   it as a standard function, it does not run.** The permitted set is bounded
   by sqlglot's own registry rather than by anyone's imagination.

   That shape is right, and its first cost estimate was wrong. "Only
   `age(...)`" was understated: review measured **eighteen** legitimate
   functions refused — roughly one in five of what a model emits for an
   ordinary question — clustered exactly where questions cluster, in dates
   and JSON. Seventeen are now named in `_ALLOWED_UNKNOWN_FUNCTIONS`, each
   checked against the three promises that list makes; `setseed` stays
   refused because it mutates session state and answers no question here. An
   allow-list is only honest if entries get added as they surface.

**Control 1 is verified, not assumed.** `verify_readonly_dsn` runs before the
first query on a DSN and refuses to serve `/ask` unless the connection is
genuinely restricted: not a superuser, unable to write to an allowlisted
table, unable to read an excluded one. Pasting the admin DSN into
`ADSIGNAL_POSTGRES_READONLY_DSN` previously deleted control 1 in silence,
which is the failure mode that turns a validator bug into an exfiltration.
A control nobody verifies is a control that might not be there.

The allowlist is exactly `fct_campaign_daily`, `dim_campaign`,
`anomaly_predictions`, `model_metrics`, `pipeline_runs`, `narrative_runs`
and `drift_reports` (`warehouse/ask_schema.py`). `model_artifacts` and
`entity_embeddings` are excluded: one holds pickled estimator bytes, the
other raw vectors, neither can contribute to an answer, and either could
return megabytes of binary.

**The prompt is a quality measure, not a control.** It asks for the same
shape the validator enforces, because a model that writes acceptable SQL
produces answers rather than refusals. A model that ignores every word of it
produces a refusal, not a breach. An LLM instructed to emit only SELECT is
not a security control; a read-only database role is.

Question-to-SQL is a **sibling** contract to `NarrativeProvider`, not an
extension of it. `NarrativeRequest` is a closed dataclass of precomputed
statistics, and its docstring gives the reason — the prompt is built from it,
so nothing unexpected can reach a provider. A free-text question shares none
of its fields, so widening it would have weakened that guarantee for the
three providers that already implement it. `narrative/sqlgen.py` defines
`SqlProvider` and `SqlQuestion` instead; the Gemini transport, the key
redaction and the circuit breaker are shared.

**The model name is a setting, not a literal.** `ADSIGNAL_GEMINI_MODEL`
(`Settings.gemini_model`, default `gemini-3.5-flash`) is passed to both
`GeminiProvider` and `GeminiSqlProvider`; neither carries a default of its
own, so they cannot drift onto different models and neither holds a stale
name. The previous hardcoded `gemini-2.0-flash` was retired by Google and
answered 404 to a valid key — a hardcoded model name is a dated assertion
about someone else's product. `config.py` records how to re-derive the
current list from the ListModels endpoint.

**No key configured is a state, not an error.** With
`ADSIGNAL_GEMINI_API_KEY` unset — which is how this repository runs today —
`/ask` answers 200 with `refusal: "ask is not configured …"` and
`rule: "not_configured"`, which the dashboard renders. The same is true when
no read-only DSN is set. Neither is ever a 500, and neither is ever a
fabricated answer.

**The endpoint answers 200, 429 or 422, and nothing else.** A rate-limit
refusal and a malformed body are the two status codes; every other outcome —
a model failure, a refused query, a database error, and anything nobody
anticipated — is a 200 carrying the same eight fields. That last clause needed
a catch-all around the answer path to be true rather than aspirational, and
the failure that proved it was an ordinary one: SQL truncated at a token
limit raises `sqlglot.TokenError`, a *sibling* of `ParseError` rather than a
subclass, which escaped the validator and became a 500. A deeply nested query
raises `RecursionError`, which is not a sqlglot error at all.

The generated SQL is returned to the caller on purpose. A reader who can see
the query can judge the answer; one who cannot has to trust it.

**What these controls do not cover.** Four residual exposures, all accepted.
The class of defect behind the pinned settings is open-ended: a future
Postgres or sqlglot version could introduce another setting the parser
assumes, and nothing here detects that automatically — only the five known
ones are pinned. `FETCH FIRST n ROWS WITH TIES` is accepted with its bound
read as `n` and can return more than `n` rows, since ties are returned in
full; `fetchmany` still stops at 200 and `MAX_RESULT_BYTES` still bounds the
answer, so the effect is that "at most n" is really "at most 200". And two of
longer standing:
`psycopg_pool` logs a failing connection's host, port and database user (never
the password), so an operator's log can name the read-only role and the host
it reached; and `ensure_readonly_role` must send the role's password to
Postgres as part of a `CREATE ROLE`/`ALTER ROLE` statement, which the server's
own `log_statement` would record if it were enabled. Neither is reachable from
an API response. The second is why that DDL does not go through
`apply_schema`, whose memo would otherwise retain the password in this
process for the life of the run.

---

## 6. Circuit breakers: per-process, and they reset every night

This is the disclosure that matters most, because the code implies a property
the deployed topology does not have.

`NarrativeChain` gives each provider its own `CircuitBreaker`
(`src/adsignal/narrative/breaker.py`): a provider that fails
`failure_threshold` times consecutively is skipped without being called until
its cooldown elapses. `src/adsignal/pipeline/definitions.py` builds the chain
**at module import** — `build_chain(_settings)` is evaluated once when the
module loads — and the comment there explains why: "state rebuilt per run is
state that never fires."

That gives one chain, and therefore one set of breakers, **per process**.

`dagster dev` launches each run in a fresh subprocess. A fresh subprocess
imports `definitions.py` again, which calls `build_chain` again, which
constructs new `CircuitBreaker` objects with `_consecutive_failures = 0` and
`_opened_at = None`. So in the deployed nightly topology, breaker state does
not survive from one run to the next, and `is_open()` is effectively always
`False` at the start of every run. Since `daily_narrative` calls `generate`
exactly once per run, a provider is attempted at most once per run anyway —
which means the breakers, in production, never skip anything.

**What actually caps a dead provider's cost is the chain's total time
budget.** `ADSIGNAL_NARRATIVE_BUDGET_SECONDS` (45 s by default,
`Settings.narrative_budget_seconds`) gates which providers are *entered*. It
is not a hard wall-clock bound — `provider.generate` is a blocking call that
is never cancelled, so a provider entered just under the deadline still runs
to its own completion, and the real bound is the budget plus that last
provider's stall. But it is the mechanism that stops four stalled backends
from holding the nightly job for the sum of their per-provider timeouts. A
provider that answers after the budget has run out is recorded as a failure.

The breakers are still correct and still worth having: they are the right
design if the chain is ever hosted in a long-lived process (a worker, or the
API), and `NarrativeChain` is written so that it works there. They are simply
not load-bearing under `dagster dev`. Two tests in `tests/test_ml_assets.py`
certify in-process sharing; their docstrings now say explicitly that this is
what they pin.

Belt and braces beneath all of it: the chain always ends in a local
`StubProvider`, exempt from the budget, so a deployment with no API keys at
all renders placeholder narrative text rather than failing.

---

## 7. GitHub Actions as the production scheduler

`src/adsignal/pipeline/definitions.py` declares a Dagster
`ScheduleDefinition` at `0 4 * * *` UTC, which is what runs under a local
`dagster dev`. For the deployed system the scheduler is GitHub Actions cron,
running the pipeline on an ephemeral runner and then
`python -m adsignal.cli.load_marts` to push the built marts into hosted
Postgres.

This is a deliberate free-tier choice, not an oversight. A Dagster daemon
needs a process that is always up; free web hosting does not provide one (see
§9 — the API itself is allowed to sleep). Actions gives a scheduler, a
runner, logs and retries at no cost, and the job is a once-nightly batch with
no latency requirement, which is the shape Actions is good at. The spec
sanctions it.

The cost is honest: the runner is ephemeral, so anything not loaded into
Postgres before the job ends is gone — which is exactly why
`cli/load_marts.py` exists and why the last pipeline commit calls it "the
step that makes the result outlive it." Actions cron is also best-effort;
runs can be delayed under load, and scheduled workflows are disabled on
inactive repositories.

**Status:** the topology is decided and the loader entry point is committed
and tested (`tests/test_load_marts.py`). The cron workflow itself is not yet
in `.github/workflows/` — the two workflows there today are `ci.yml` and
`eval.yml`, both triggered by push and pull request. Nothing in this
repository is scheduled by cron at the time of writing.

---

## 8. Memory: 409-440 MB measured against a 512 MB ceiling

`scripts/measure_memory.py` starts the app, issues `/health`, issues one
`/search`, issues one `/score`, and reports peak RSS.

**This document deliberately quotes a range rather than a figure.** Five runs
of *identical* code at the same commit, same image, same command produced:

```
408.9   423.6   427.6   438.7   440.2      (MB peak RSS)
```

A 31.3 MB spread on code that did not change. The stages isolate where it
comes from: `create_app` and `/health` are stable to within ~1.3 MB across all
five runs, and the entire spread appears once `/search` and `/score` have run.
That is onnxruntime's arena allocator, which sizes its pools from the
allocation pattern it happens to see, not from anything the application
controls.

The honest consequence is that **any single memory number from this project is
one sample from a ~30 MB-wide distribution.** An earlier revision of this
document quoted 404.1 MB to four significant figures; that figure was real,
and it was also just the low end of a band. The worst observed run leaves
72 MB of headroom under the free tier's 512 MB.

The two fitted estimators `/score` unpickles cost ~16 MB on top of `/search`
— small next to the embedder, because an `IsolationForest` of 200 shallow
trees and two gradient-boosted ensembles are, unlike a transformer, mostly
split thresholds. `api/model_loader.py` unpickles them one at a time and drops
each raw `bytea` blob's last reference before unpickling the next, so the two
serialized blobs and the two fitted estimators are never all resident at once.

The script fails the run if `/score` answers anything but 200, because a peak
measured without the models loaded is an understatement rather than a pass.

If the ceiling is ever breached in production, the first lever is to serve
`/score` behind a feature flag — the endpoint already degrades to a named 503
rather than crashing — and the second is the embedder fallback described
below.

What did *not* fit was MLflow. `ml/scoring.py` reaches `load_models` through
`ml/registry.py`, which is also the module that logs training runs; a
module-scope `import mlflow` there costs a measured ~136 MB of resident
memory, which alone would have put the API over the ceiling. That import now
lives inside `log_training_run`, the one function that needs it, and
`test_loading_the_scoring_layer_does_not_import_mlflow` holds it there. The script exits non-zero above 85% of the budget, which
makes the headroom *checkable* on demand — but nothing invokes it: `git grep
measure_memory` over `.github/` and `tests/` returns nothing. It is a manual
script, so this is a measurement someone has to remember to take, not an
invariant CI enforces. Wiring it in would need a container step in CI and is
deliberately left out of this task.

Almost all of that is the embedder: ~90 MB of int8 ONNX weights plus
onnxruntime's arenas plus the interpreter. It is loaded **lazily** —
`EmbeddingModel.__init__` stores a name and a lock, and `TextEmbedding` is
constructed on first `embed()` — and the first load is serialised behind that
lock under double-checked read, because `/search` is a sync `def` and
Starlette dispatches it to the anyio worker threadpool, where an
unsynchronised lazy attribute would let several concurrent cold requests each
construct their own ~90 MB model.

### The rate limiter is a memory control, not just an abuse control

In `src/adsignal/api/routes/search.py` the ordering is explicit and worth
reading:

```python
if not rate_limiter.allow(client_key(request, settings.trusted_proxy_hops)):
    raise HTTPException(status_code=429, detail="rate limit exceeded")

hits = semantic_search(...)  # this is where the ONNX forward pass happens
```

The limiter check runs **before** anything touches the model. A refused
request costs a token-bucket lookup and returns 429; it never enters
`semantic_search`, never calls `model.embed_one`, and — on a cold process —
never triggers the lazy weight load. So `SEARCH_RATE_LIMIT_CAPACITY` (5,
refilling at 1/s) is what bounds how many embedding calls can be in flight at
once, and therefore contributes to bounding peak memory. Raising `/search`'s
limit would allow more concurrent inference and push peak memory toward the
512 MB ceiling.

To be precise about what is and is not known: **no controlled experiment
comparing concurrent to serialized load exists in this repository.** Every
figure above is a single-request measurement, and as the range at the top of
this section shows, even those vary by ~30 MB between runs. The claim about
the limiter is about ordering in the code — which is verifiable by reading it
— not about a measured delta between load profiles. Nobody should quote a
concurrency number from this document, because there isn't one.

If the ceiling is ever breached, the fallback is to drop the embedder from
the API process entirely and serve `/search` from precomputed vectors only.

### A measurement trap

`scripts/measure_memory.py` must be run **inside the container**, for two
independent reasons.

The first is a unit bug waiting to happen. `ru_maxrss` is reported in
kilobytes on Linux and in *bytes* on macOS, and `peak_rss_mb()` divides by
1024 unconditionally. On Linux that is kilobytes → MB, which is right. On a
macOS host it is bytes → kilobytes printed under an MB label, so the figure
comes out **1024× too large**:

```
true peak                412.0 MB
macOS ru_maxrss      432013312 bytes
printed by the script   421888.0 "MB"
```

Too large, not too small — worth being precise about, because the failure is
loud rather than quiet. Nobody believes a 421,888 MB process, so the mistake
announces itself. A reader who expected an implausibly *small* number would be
hunting for the wrong symptom, and might accept a plausible-looking wrong
figure instead.

The second reason is that a host-side measurement misses the container's
interpreter, onnxruntime's arenas and the loaded model, so even on Linux it
answers a different question than the one the 512 MB limit asks. The script's
docstring originally gave only this second reason; it now states both, so a
reader who follows the citation finds the unit trap there too.

---

## 9. Known limits

Stated plainly, because a reviewer will hit all of these.

**The API sleeps.** On the free hosting tier the service is suspended after
15 minutes without traffic. The first visit after an idle period pays a cold
start of roughly a minute before the first response. This is the hosting
provider's documented free-tier behaviour, not something measured here.
Nothing in the code works around it; a keep-alive ping would trade the
platform's cost for ours and was not worth it for a portfolio deployment.

**The free Postgres is 0.5 GB.** That is the storage ceiling the marts, the
predictions, the model metrics, the drift reports and the pgvector embeddings
all share. It is why only aggregated daily marts are loaded — the raw event
stream, ten million rows, never leaves DuckDB and Parquet.

**Connection pooling is per-process only.** `src/adsignal/warehouse/pool.py`
keeps `_POOLS` as a module-level dict keyed by DSN, refcounted through
`acquire_pool`/`release_pool`. Every process gets its own pools. Two web
workers, or a web worker and a pipeline run, do not share connections and
their `pool_max_size` budgets add up against the hosted database's own
connection ceiling. On a single free-tier instance that is fine; it would not
survive horizontal scaling without an external pooler.

**The nightly pipeline has no scheduler committed yet.** See §7.

**Breaker state resets every night.** See §6.

**`/ask` is only as good as one Gemini call.** There is no retry across
providers, no repair loop on a refused query and no few-shot examples: a
question the model answers with unacceptable SQL comes back as a refusal
naming the rule, and the user rephrases. The refusal shows the rejected
query, so rephrasing is informed rather than blind.

**The models recover an encoded signal.** `anomaly_cause` is assigned by the
synthetic generator from deterministic effect multipliers, so the classifier
is not discovering structure in the world. What this demonstrates is the
engineering around a model — leak-free feature windows, temporal splits,
versioned runs, a CI gate, drift monitoring — not a research result.

**The test suite cannot be run concurrently with itself.** Every test DSN in
`tests/conftest.py` is a fixed database name on a shared Postgres server, so a
second `pytest` process truncates the first run's tables underneath it. The
six `ADSIGNAL_*_TEST_DSN` overrides are the seam a per-worker database suffix
would hang off — deliberately not built here; see `CONTRIBUTING.md`.

**A fixture that performs production's job makes the test pass and the
product broken.** `load_dim_campaign` was written, tested, and read by the
API's campaign and anomaly queries, but no Dagster asset ever called it, so
every real pipeline run left `dim_campaign` empty and the dashboard rendered
`cmp_0058_006` where a campaign name belonged. The dimension tests passed
throughout, because `tests/conftest.py` calls `load_dim_campaign` itself to
build its fixtures — validating the loader in isolation from the only thing
meant to invoke it. Fixed in commit `b35217f`: `campaign_daily_mart` now calls
the loader, and the regression test asserts against the database after a
materialization, never calling the loader itself, and checks that every fact
campaign *resolves to* a dimension row rather than merely that rows exist — a
`count > 0` check would have passed against stale fixture rows and certified
the bug.
