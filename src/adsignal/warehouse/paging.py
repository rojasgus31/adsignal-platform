"""The pagination contract the serving reads and the routers share.

`run_log.py` and `serving_reads.py` each used to define a module-level
`DEFAULT_PAGE_LIMIT`, with different values (50 and 100), plus a
`MAX_PAGE_LIMIT` of 500 apiece. Two constants with one name and two values is
a trap: a router importing "the" default got whichever module it happened to
import from. The defaults live here instead, named for the surface each one
belongs to, so the difference is visible at every call site.

They stay in `warehouse/` rather than in a router because `api/` depends on
`warehouse/`, never the reverse: the layer that enforces a bound is the layer
that declares it.
"""

# The ceiling every paginated reader and endpoint shares. An unbounded
# collection is a denial-of-service lever: one response otherwise grows with
# the table, and clients have no pagination contract to code against.
MAX_PAGE_LIMIT = 500

# `serving_reads.read_model_metrics` and `read_anomaly_predictions`, and so
# `GET /models` and `GET /anomalies`.
DEFAULT_SERVING_PAGE_LIMIT = 100

# `run_log.read_runs`, and so `GET /operations`. Half the serving default:
# one operations row is one asset materialization, and a page of 50 already
# covers several nightly runs. The dashboard client's `operations()` default
# is 50 to match.
DEFAULT_RUN_PAGE_LIMIT = 50
