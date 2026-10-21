# Site Deployment Checklist

## Site survey (six weeks before go-live)

- Measure aisle widths, floor grade, and shelf heights against the safety requirements.
- Run a Wi-Fi heat map: signal must be stronger than -67 dBm in every aisle, with roaming between access points under one second.
- Identify locations for charging docks (one dock per four robots) and packing stations.
- Confirm power: each dock needs a dedicated 20 A circuit.

## Network (four weeks before)

Create a dedicated VLAN for robots and the Relay site server, allow outbound HTTPS to the Helios cloud, and open the site-to-cloud VPN. The site server needs a static IP and NTP access.

## Installation (two weeks before)

Install docks and light curtains, apply floor markings for robot lanes and human-only zones, and build the site map in the Relay map editor. Run a 48-hour simulation with Relay's virtual robots against the customer's real order history.

## Go-live

Start with a pilot of five robots for one shift, then scale up by 25% per day while watching pick rate and intervention count. The customer's site lead signs the go-live checklist, which triggers the start of the support contract.

## Training

Train site operators on the Relay dashboard, robot safety, and basic recovery (clearing a blocked robot, swapping a tote). Training records are stored in the customer's account in the Helios console.
