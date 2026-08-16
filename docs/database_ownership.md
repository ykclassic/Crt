# Nexus Database Ownership Contract

## Purpose

Nexus uses two SQLite databases with explicit write ownership:

| Database | Owner | Governance access | Purpose |
|---|---|---|---|
| `nexus_signals.db` | Intelligence Pipeline | Read-only | Canonical signal-generation record |
| `nexus_governance.db` | Governance | Read/write | Signal outcomes, engine state, audit state |

## `nexus_signals.db`

The intelligence pipeline is the sole writer.

Pipeline responsibilities:

1. Initialize or safely migrate the schema.
2. Generate and insert signals.
3. Persist the database to the repository.
4. Publish the current database as the `nexus_signals_db` GitHub Actions artifact.

Governance must never execute `INSERT`, `UPDATE`, `DELETE`, `ALTER`, or `DROP` against this database.

`alert_monitor.py` explicitly opens this database with SQLite `mode=ro`.

## `nexus_governance.db`

Governance owns this database.

It stores:

- signal evaluation outcomes (`PENDING`, `WIN`, `LOSS`, `ERROR`)
- evaluation timestamps and evidence
- engine governance state (`LIVE`, `RECOVERY`, `DISABLED`)
- aggregate performance metrics

The governance workflow commits this database to Git so state survives workflow runners.

## Signal lifecycle

The raw signal row is not mutated when TP/SL is detected.

Instead:

```text
nexus_signals.db
    signal #1234
        |
        | read-only
        v
alert_monitor.py
        |
        v
nexus_governance.db
    signal_evaluations
        outcome = WIN / LOSS / PENDING / ERROR
```

This prevents governance activity from corrupting the pipeline's source record.

## Engine governance lifecycle

Performance audit consumes finalized evaluations:

```text
WIN/LOSS evaluations
        |
        v
performance aggregation
        |
        +--> engine_governance
        |
        +--> performance.json
```

The state machine is:

```text
LIVE --(< kill threshold)--> RECOVERY
RECOVERY --(>= recovery threshold)--> LIVE
```

A minimum sample size is required before a state transition.

## Workflow ownership

### `alert_cron.yml`

Sole writer of `nexus_signals.db`.

### `nexus-governance.yml`

Reads the latest successful pipeline artifact and writes only:

- `nexus_governance.db`
- `performance.json`

### `weekly_report.yml`

Reads `nexus_governance.db` and produces `weekly_performance_report.csv`.

The former duplicate `nexus-monitor.yml` workflow is intentionally removed.

## Concurrency

All workflows that can write to `main` use the shared GitHub Actions concurrency group:

```yaml
concurrency:
  group: nexus-main-writer
  cancel-in-progress: false
```

This serializes repository state mutations and reduces the risk of competing commits.

## Recovery rule

Legacy signal schemas are never dropped. Known legacy columns (`engine` and `pair`) are mapped to the canonical `engine_id` and `symbol` fields. Unknown incompatible legacy tables are retained under a `signals_legacy_N` name for manual recovery.
