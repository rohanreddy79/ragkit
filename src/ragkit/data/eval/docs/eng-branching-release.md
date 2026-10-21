# Branching and Release Trains

Helios uses trunk-based development. All work happens on short-lived branches off `main` and merges back within a few days. Long-lived feature branches are not allowed; use feature flags to hide incomplete work.

## Release trains

A release branch is cut from `main` every second Wednesday at 10:00 PT by the release captain. Releases are named by year and ISO week, for example `2026.14`. The release branch gets a week of staging soak and customer-preview testing before it is promoted to production the following Wednesday.

## Hotfixes

A hotfix is a change that must reach production before the next train. Open a PR against `main`, get it merged, then cherry-pick it to the current release branch with the `hotfix` label. The on-call engineering manager approves hotfix promotions. Hotfixes still go through canary.

## Versioning

Services use the release name as their version. Libraries published to the internal package registry use semantic versioning; a breaking change bumps the major version and requires a migration note in the changelog.

## Release captain duties

The release captain rotates weekly among senior engineers. Duties: cut the branch, post the release notes draft in #releases, confirm staging soak results, run the promotion, and update the release calendar.
