# Incident Severity Levels

## Definitions

- SEV1: a customer-facing outage, robots stopped across a site, or a data loss or security breach. All hands; status page updated.
- SEV2: major degradation, such as a 10% or higher error rate on the Beacon API or a single site running at reduced throughput. Status page updated.
- SEV3: minor degradation or an internal tool outage with a workaround. No status page update.
- SEV4: cosmetic bugs or issues with no customer impact, tracked as normal bugs.

## Response expectations

SEV1 requires acknowledgement within 15 minutes and an incident commander within 30 minutes; executive stakeholders are paged. SEV2 requires acknowledgement within 30 minutes and updates every hour. SEV3 is handled during business hours within one day. The incident commander owns communication; engineers own mitigation.

## Communication

Customer-facing updates go on the status page and are approved by the incident commander. Internal updates go to #incidents. Customers on the Premium or Enterprise support tier receive a direct email from their account manager for SEV1 and SEV2.

## Postmortems

Every SEV1 and SEV2 gets a blameless postmortem published within five business days. The postmortem records the timeline, root cause, contributing factors, and action items with owners and due dates. Action items are tracked in the RELIAB Jira project and reviewed weekly until closed. SEV3 postmortems are optional but encouraged for recurring issues.
