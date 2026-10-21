# On-Call Runbook

## Rotation

Every production service has a weekly on-call rotation in PagerDuty with a primary and a secondary engineer. Handoff happens on Monday at 10:00 PT during the weekly operations sync, where the outgoing primary summarises open incidents and noisy alerts. Swaps are arranged in PagerDuty and announced in #oncall.

## Responding to a page

The primary acknowledges a page within 15 minutes. If the page is not acknowledged within 15 minutes, PagerDuty escalates to the secondary; after 30 minutes, it escalates to the engineering manager. Acknowledge first, then investigate. Post a first update in #incidents within 30 minutes of acknowledging, even if the update is "still investigating".

## Declaring an incident

If a page affects customers or more than one site, declare an incident with `/incident declare` in Slack, which creates a channel, a status page draft, and assigns you as incident commander until someone else takes over. Severity levels and comms expectations are in the incident severity guide.

## Common procedures

- Relay API returning 5xx: check the `relay-api` Grafana dashboard for pod restarts, then run `relay-deploy rollback` if the last deploy was within two hours.
- Fleet heartbeat gaps: confirm the site VPN tunnel status before suspecting the robots.
- Queue backlog alerts: scale the task-allocator deployment to six replicas and open a ticket.

## Compensation

On-call engineers receive $350 per primary week and $150 per secondary week, paid monthly. Time spent on incidents outside working hours can be taken back as time off in lieu with manager agreement.
