# Secrets Management

## Where secrets live

All credentials, API keys, and certificates are stored in Keyring, our Vault deployment. Services read secrets at startup through the Keyring agent sidecar; they are never baked into container images or committed to a repository. The pre-commit hook and the CI secret scanner block commits that look like keys, and a blocked commit should be treated as a potential leak even if it was a test value.

## Rotation

Secrets rotate every 90 days at most. Database credentials and Beacon signing keys are rotated automatically by Keyring; third-party API keys are rotated by the owning team, who receive a reminder 14 days before expiry. A secret that may have been exposed is rotated immediately and reported to Security.

## Access

Engineers get read access to a service's secrets only through the service's Keyring policy, granted via an access request. Production secrets are not readable from laptops; use the `keyring exec` wrapper in a staging environment for debugging.

## Break-glass

If Keyring is unavailable during an incident, the on-call engineering manager can approve a break-glass read from the sealed backup, which pages the Security on-call and expires after four hours. Every break-glass use gets a postmortem entry.

## Local development

Use `keyring dev` to populate a local `.env` with sandbox credentials. Never copy production values into `.env` files.
