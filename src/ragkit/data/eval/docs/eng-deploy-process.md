# Deployment Process

## Pipeline

Every merge to main runs the CI pipeline: lint, unit tests, integration tests against a disposable Postgres, and a container build pushed to the registry. A green build on main is automatically deployed to the staging environment. Production deploys are triggered manually with `relay-deploy promote <build>` by the owning team.

## Canary and rollout

Production deploys start with a canary at 5% of traffic for 30 minutes. The deploy tool watches error rate and p99 latency; if either breaches the service's SLO budget, the canary is rolled back automatically. After a healthy canary, the rollout proceeds in stages of 25%, 50%, and 100%, with a 10-minute soak at each stage.

## Deploy windows

Deploy between 09:00 and 16:00 PT on Monday through Thursday. No production deploys on Friday after 12:00 PT or on the day before a company holiday, except for hotfixes approved by the on-call engineering manager. Robot firmware is deployed to customer sites only in agreed maintenance windows.

## Rollback

`relay-deploy rollback <service>` returns the previous build within about two minutes. Roll back first and investigate afterwards. Database migrations are never rolled back automatically; follow the migrations guide.

## Feature flags

New behaviour ships dark behind a Switchboard feature flag. Flags are enabled per customer or per site, and every flag has an owner and a removal date. Flags older than 90 days show up on the weekly flag-hygiene report.
