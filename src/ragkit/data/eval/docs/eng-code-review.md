# Code Review Guidelines

## Approvals

Changes to `fleet-core` and `beacon-api` require two approvals, one of which must come from a CODEOWNERS entry. All other repositories require one approval. The author may not approve their own change, and approvals are dismissed when new commits are pushed.

## Turnaround

Reviewers respond to a review request within one business day. If you cannot review in time, say so in the pull request and suggest another reviewer. Authors should keep pull requests under 400 changed lines; larger changes should be split into a stack of smaller PRs.

## What to look for

Correctness first, then tests, then readability. Comment on behaviour, not style; style is enforced by the formatter and linter in CI. Prefer questions over directives for anything that is a matter of taste. Mark comments as "nit" when they should not block a merge.

## Commit messages

Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`. The PR title becomes the squash commit message, so write it as a complete sentence describing the change and link the Jira ticket.

## Merging

PRs are squash-merged by the author after approval and a green CI run. Do not merge a PR with a failing or skipped required check. Revert first if a merged change breaks main; discuss afterwards.
