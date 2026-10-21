# Atlas Firmware 4.2 Release Notes

Firmware 4.2 ships to customer sites in the Q2 maintenance windows. It supports hardware revisions B and C. Rev A units remain on the 3.9 long-term support line.

## Highlights

- Docking: a new approach controller halves failed docking attempts in internal testing and tolerates dock misalignment of up to 3 cm.
- Battery estimation: the state-of-charge model now accounts for pack temperature, fixing the premature low-battery returns reported at cold-storage sites.
- Picking: wrist-camera item verification can be enabled per SKU and rejects picks whose barcode does not match the task.
- Safety: protective stop recovery now requires an operator acknowledgement on the robot's screen.

## Known issues

- On Rev B units, Wi-Fi roaming between access points may take up to 4 seconds; a fix is planned for 4.2.1.
- The dashboard may show a stale battery value for up to 30 seconds after a robot leaves the dock.

## Upgrade procedure

Upgrades are staged through Relay: upload the firmware bundle in the Relay dashboard, select a pilot group of at most five robots, run them for one shift, then roll out to the rest of the fleet. Robots must be docked with at least 50% charge to begin an upgrade. Rolling back to 4.1 is supported within the same maintenance window.

## Support

Report firmware issues through the support escalation process with the robot serial number and the diagnostic bundle exported from the Relay dashboard.
