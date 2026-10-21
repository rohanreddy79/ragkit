# Database Migrations

Relay and Beacon store state in PostgreSQL. Schema changes are written as migration files and applied with the `mig` tool, which records each migration in the `schema_migrations` table.

## Expand and contract

Every migration must be backwards compatible with the version of the service currently running, because deploys are rolling and migrations run before the new code starts. Use the expand/contract pattern: first add the new column or table (expand), deploy code that writes to both old and new, backfill, switch reads, and only then drop the old column (contract) in a later release. Never rename a column in place.

## Destructive changes

Dropping a table or column, or changing a column type, may not ship in the same release as the code that stops using it. Destructive migrations require review from the data-platform team and a note in the release notes.

## Backfills

Backfills run as batched jobs of at most 10,000 rows per transaction with a short sleep between batches, so they do not hold locks or saturate replication. Large backfills are scheduled outside peak hours, between 05:00 and 07:00 PT.

## Testing

Migrations run in CI against a disposable Postgres container, and the pipeline fails if a migration takes longer than 30 seconds on the seeded dataset. Index creation uses `CREATE INDEX CONCURRENTLY` so it never blocks writes.
