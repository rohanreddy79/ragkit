# Relay Fleet Manager

Relay is the orchestration software that runs an Atlas fleet at a customer site. It receives orders from the customer's warehouse management system (WMS), breaks them into pick tasks, assigns tasks to robots, and manages traffic so robots do not block each other.

## Task allocation

The allocator scores every pending task against every idle robot using travel distance, battery level, and tote capacity, then assigns in a batch every 500 ms. Tasks are re-planned when a robot goes offline. Customers can set per-zone priorities, for example to clear a cold-storage zone first.

## Traffic management

Relay reserves aisle segments ahead of each robot and resolves conflicts with a priority rule: the robot carrying more totes yields. Dead ends and one-way aisles are configured in the site map editor. Human-only zones are excluded from routing entirely.

## Scale and deployment

A single Relay instance supports up to 500 robots and 20,000 open tasks per site. Relay runs on-premises on the site server for latency and resilience, syncing orders and metrics to the Helios cloud. If the cloud link drops, the site keeps operating from local state for up to 72 hours.

## Integrations

Relay integrates with Manhattan, Blue Yonder, and SAP EWM out of the box, and any WMS can integrate through the Beacon API. Site operators use the Relay dashboard to watch robots, pause zones, and inspect task history.
