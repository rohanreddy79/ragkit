# Beacon API Overview

Beacon is the public REST API for Relay. Customers use it to submit orders, query robot status, manage zones, and receive events.

## Authentication

Requests carry an API key in the `Authorization: Bearer` header. Keys are issued per site in the Helios console and can be scoped read-only or read-write. Keys never expire automatically but can be revoked instantly; rotate them at least yearly.

## Rate limits

Each key is limited to 600 requests per minute, with bursts of up to 100 requests per second. Responses include `X-RateLimit-Remaining` and `Retry-After` headers; a 429 response means back off and retry with exponential delay.

## Versioning

The current version is v2, reachable at `/v2/`. Breaking changes ship as a new version; v1 remains available for 12 months after a successor is released. Non-breaking fields may be added at any time, so clients must ignore unknown fields.

## Core endpoints

- `POST /v2/tasks` creates a pick task; `GET /v2/tasks/{id}` returns its status.
- `GET /v2/robots` lists robots with battery, position, and current task.
- `GET /v2/zones` and `PATCH /v2/zones/{id}` manage zone priority and pause state.

## Pagination and webhooks

List endpoints use cursor pagination with a `next_cursor` field and a maximum page size of 200. Webhooks deliver `task.completed`, `task.failed`, and `robot.offline` events with an HMAC-SHA256 signature in the `X-Beacon-Signature` header; delivery retries follow exponential backoff for up to 24 hours.

## Sandbox

Every customer gets a sandbox site with simulated robots; sandbox keys start with `sk_test_`.
