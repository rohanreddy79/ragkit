# Access Requests and Reviews

## Principles

Access follows least privilege: people get the minimum access needed for their current role, for the shortest reasonable time. Role-based groups in Okta grant baseline access automatically when someone joins a team; anything beyond that is requested individually.

## Requesting access

Open a HelpDesk access request naming the system, the permission level, and the business reason. Requests need approval from your manager and from the system owner listed in the service catalogue. Standard requests are fulfilled within two business days; production database and admin requests need a Security review and may take longer.

## Privileged access

Admin and production access is time-boxed: it expires automatically after 90 days unless renewed, and every use is logged. Emergency access during an incident uses the break-glass procedure described in the secrets management guide.

## Access reviews

System owners review who has access to their system every quarter. Managers review their reports' access when someone changes role. Access not confirmed within the review window is removed automatically. Leavers lose all access at the end of their last day through the Okta offboarding workflow.

## Contractors and vendors

Contractor accounts expire with the contract end date recorded in Workday. Vendor access to production is permitted only through a monitored session with a Helios engineer present.
