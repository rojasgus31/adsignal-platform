"""Deployment bootstrap: create the read-only role `/ask` runs its SQL as.

`warehouse.ask_reads.ensure_readonly_role` builds `adsignal_ro` — SELECT on
the seven allowlisted tables, `REVOKE ALL` on `model_artifacts` and
`entity_embeddings`. Until this module existed, nothing in `src/` called it.
A deployment that created the role's login by hand, set
`ADSIGNAL_POSTGRES_READONLY_DSN` and started the API passed the privilege
probe and then refused every question with `rule: "database_permission"`,
because no grant had ever been applied.

This is a command, not a request-time action, for one reason: applying the
grants needs the **admin** DSN. Doing it per request would mean the serving
process holds admin credentials, which is the opposite of what the read-only
role exists for. Run it once per deploy, from wherever the admin DSN already
lives (the release step, a migration hook, a laptop):

    python -m adsignal.cli.bootstrap_readonly_role

It takes no arguments. Both DSNs come from the environment through
`Settings`, and the role name and its password are read out of
`ADSIGNAL_POSTGRES_READONLY_DSN` itself — the same string the API connects
with — so the grants cannot be applied to a role the API does not use.

Nothing here prints, logs or returns the password, on success or on failure.
The report names the role, the granted tables and the revoked ones, all of
which are public facts about the schema.
"""

import sys
from dataclasses import dataclass

from adsignal.warehouse.ask_reads import (
    EXCLUDED_TABLES,
    QueryRefused,
    ensure_readonly_role,
    readonly_credentials,
    verify_readonly_dsn,
)
from adsignal.warehouse.ask_schema import ALLOWED_TABLES
from adsignal.warehouse.pool import close_all_pools


@dataclass(frozen=True)
class BootstrapReport:
    """What one bootstrap did. Carries table and role names — never a DSN or password."""

    role: str
    granted_tables: tuple[str, ...]
    revoked_tables: tuple[str, ...]


def bootstrap(admin_dsn: str, readonly_dsn: str) -> BootstrapReport:
    """Apply the read-only role's grants, then prove the DSN really is restricted.

    Idempotent, because `ensure_readonly_role` is: an existing role has its
    password reset and its grants reapplied, so a deploy hook can run this
    every time.

    The verification afterwards is the same check `/ask` makes before its
    first query (`verify_readonly_dsn`): not a superuser, cannot write to an
    allowlisted table, cannot read an excluded one. Running it here means a
    deploy fails at the deploy rather than at the first question.

    Raises `QueryRefused` when that verification fails and `ValueError` when
    `readonly_dsn` carries no user or no password.
    """
    role, password = readonly_credentials(readonly_dsn)
    ensure_readonly_role(admin_dsn, password, role)
    verify_readonly_dsn(readonly_dsn)
    return BootstrapReport(
        role=role,
        granted_tables=tuple(sorted(ALLOWED_TABLES)),
        revoked_tables=tuple(EXCLUDED_TABLES),
    )


def main() -> int:
    """`python -m adsignal.cli.bootstrap_readonly_role`. 0 on success.

    Exits 2 when `ADSIGNAL_POSTGRES_READONLY_DSN` is unset or unusable —
    there is nothing to grant to, and a silent success would leave the same
    bug this command exists to fix. Exits 1 when the grants were applied but
    the resulting connection did not verify as the restricted role.
    """
    from adsignal.config import get_settings

    settings = get_settings()
    if not settings.postgres_readonly_dsn:
        print(
            "ADSIGNAL_POSTGRES_READONLY_DSN is not set; "
            "set it to the adsignal_ro DSN the API will use, then run this again",
            file=sys.stderr,
        )
        return 2

    try:
        report = bootstrap(settings.postgres_dsn, settings.postgres_readonly_dsn)
    except ValueError as error:
        # `readonly_credentials` raises this naming the missing field only —
        # it never puts the DSN in the message, so neither does this.
        print(f"ADSIGNAL_POSTGRES_READONLY_DSN is unusable: {error}", file=sys.stderr)
        return 2
    except QueryRefused as refusal:
        print(f"the read-only role did not verify: {refusal.reason}", file=sys.stderr)
        return 1
    finally:
        # This process exists to run one bootstrap and exit; the pool
        # `verify_readonly_dsn` opened has no later user here.
        close_all_pools()

    print(f"role {report.role} verified")
    print(f"  SELECT granted on {len(report.granted_tables)}: {', '.join(report.granted_tables)}")
    print(f"  all privileges revoked on: {', '.join(report.revoked_tables)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
