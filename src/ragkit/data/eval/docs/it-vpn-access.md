# VPN and Remote Access

## Tunnel VPN

Production systems, the Relay staging environment, and customer site networks are reachable only through Tunnel, the company VPN. Install the Tunnel client from the self-service app catalogue; it is pre-installed on new laptops. Sign in with Okta and approve the push notification in Okta Verify. Sessions last 12 hours.

## Requirements

Your laptop must be enrolled in MDM and pass the device health check (disk encryption on, firewall on, OS patched within 30 days). Personal devices cannot connect. Split tunnelling is disabled for the production profile, so all traffic goes through the VPN while connected; the "office" profile allows split tunnelling for general use.

## Requesting access

Access to production networks and customer sites is granted per profile. Open a HelpDesk request with the profile name and a business reason; your manager and the network team approve it. Customer site profiles are time-boxed to the duration of the support ticket.

## Troubleshooting

If the health check fails, open the MDM agent and run "Check in"; most failures are a pending OS update. If Okta Verify does not prompt, check that the phone has network connectivity and that its time is set automatically. For anything else, open a HelpDesk ticket or ask in #it-help.

## Shared accounts

Shared VPN credentials are never issued. Contractors receive individual accounts that expire with their contract.
