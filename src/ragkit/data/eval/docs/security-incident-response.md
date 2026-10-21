# Security Incident Response

## Reporting

Report any suspected security incident, including phishing, a lost device with customer data, unexpected access to a system, or a leaked credential, to security@helios.example or with the "Report phishing" button in Outlook within one hour of noticing it. Reporting a false alarm is always fine; not reporting is not.

## Triage

The Security on-call triages reports within 30 minutes during business hours and within two hours otherwise. Confirmed incidents get a severity using the incident severity guide and a dedicated Slack channel. The CISO is notified for any incident involving customer data or production access.

## Containment and evidence

Contain first: disable the account, revoke the key, isolate the host, or block the sender. Preserve evidence before rebuilding anything; forensic images and logs are kept for the duration of the investigation and at least one year after. Do not discuss incidents outside the incident channel.

## Notification

If customer personal data may have been affected, Legal determines the notification obligations. GDPR requires notifying the supervisory authority within 72 hours of becoming aware of a breach, and affected customers are notified according to their contracts. Only Legal and the CISO communicate externally about an incident.

## Afterwards

Every confirmed incident gets a postmortem within five business days following the engineering postmortem template, plus a review of whether detection could have been faster.
