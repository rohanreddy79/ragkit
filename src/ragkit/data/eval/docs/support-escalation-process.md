# Support Escalation Process

## Tiers

Tier 1 (Support Engineers) owns the customer conversation, reproduces the issue, and resolves known problems using the knowledge base. Tier 2 (Site Reliability) handles site infrastructure, Relay configuration, and firmware upgrades. Engineering handles defects and anything requiring a code change.

## When to escalate

Escalate from Tier 1 to Tier 2 when the issue cannot be resolved from the knowledge base within one hour for a P1 or four hours for a P2. Escalate to Engineering when Tier 2 identifies a defect or when a P1 remains unmitigated for two hours.

## Required information

Every escalation ticket in the SUP Jira project must include: the site id, the robot serial numbers affected, the time window in UTC, the diagnostic bundle exported from the Relay dashboard, and the customer's own priority. Tickets missing the diagnostic bundle are sent back to Tier 1.

## Priority mapping

Customer priorities map to incident severities as follows: P1 becomes SEV1 if more than one site is affected, otherwise SEV2; P2 becomes SEV3; P3 and P4 are handled as bugs. The on-call engineer declares the incident.

## Customer communication

Tier 1 remains the single point of contact. Engineering does not email customers directly; updates flow through the ticket. Enterprise customers receive a written root-cause summary within five business days of resolution.
