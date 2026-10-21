# Logging and Observability

## Logs

Services emit structured JSON logs with `trace_id`, `service`, `level`, and `site_id` fields. Logs are shipped to Loki and retained for 30 days in hot storage and one year in cold object storage. Never log personal data, API keys, or robot camera frames; the log pipeline redacts known patterns, but redaction is a safety net, not a licence.

## Metrics and dashboards

Prometheus scrapes every service; Grafana hosts the dashboards. Each service has a standard dashboard with request rate, error rate, p50/p99 latency, and saturation. Robot fleet metrics (battery, pick rate, heartbeat age) are on the Fleet Overview dashboard.

## Tracing

Requests carry a W3C trace context header across services, and traces are sampled at 10% in production and 100% in staging. Use the trace id from a customer ticket to find the request in Tempo.

## Alerting

Alerts are defined in code alongside the service and routed to PagerDuty. Page-worthy alerts must be actionable and link to a runbook. Error-rate alerts fire when the 5-minute error rate exceeds 2%; latency alerts fire when p99 exceeds the SLO for 10 minutes.

## SLOs

Beacon API: 99.9% availability and p99 latency under 500 ms. Relay task allocation: 99.95% of tasks assigned within 2 seconds. SLO burn-rate alerts page the on-call engineer when the monthly error budget is being consumed faster than 2x.
