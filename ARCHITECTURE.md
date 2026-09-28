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

## 5. `/ask` selects a template; it does not generate SQL

The spec calls `/ask` "natural language to SQL". This implementation does not
generate SQL, and nothing generates it on the endpoint's behalf.

A keyword matcher (`_match_template` in `src/adsignal/api/routes/ask.py`)
maps a question to the *name* of one of four templates declared in
`src/adsignal/api/queries.py`. That name is validated against
`QUERY_TEMPLATES` before anything runs; an unrecognised name is rejected with
a 400 rather than executed. Each template's `sql` is a fixed literal with
`%s` placeholders, and each declared parameter is validated by an `IntParam`
with a minimum, a maximum and a default before it is bound. Executing a
template is `warehouse/template_reads.run_template`'s job, so `api/` issues no
SQL at all.

The security property here is **structural, not defensive**. This is not
injection sanitisation applied after the fact. A selector whose entire output
alphabet is four known strings cannot emit a query, so it cannot emit an
injection — there is no string-building step for a malicious input to reach.

**No model is involved in that selection today, in any configuration**, and
the reason is a contract, not an oversight. The provider interface is
`NarrativeProvider.generate(request: NarrativeRequest) -> str`
(`src/adsignal/narrative/types.py`), where `NarrativeRequest` is a closed
dataclass of precomputed statistics — campaign and anomaly counts, top
causes. It has no field capable of carrying an arbitrary free-text question.
Routing `/ask` through the narrative chain, as the plan's prose describes,
would mean changing a contract that several earlier tasks already depend on.
The keyword matcher is what ships.

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

**`/ask` answers four questions.** The template registry has four entries.
A question that matches no keyword rule returns "I don't have a named query
that answers that question yet" rather than guessing.

**The models recover an encoded signal.** `anomaly_cause` is assigned by the
synthetic generator from deterministic effect multipliers, so the classifier
is not discovering structure in the world. What this demonstrates is the
engineering around a model — leak-free feature windows, temporal splits,
versioned runs, a CI gate, drift monitoring — not a research result.
