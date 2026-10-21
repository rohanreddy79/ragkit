# Atlas Picker Robot: Specification

Atlas is Helios Dynamics' autonomous mobile picking robot for e-commerce and retail warehouses. It navigates shared aisles alongside people, picks items from shelves into totes, and delivers full totes to a packing station.

## Key specifications

- Payload: 15 kg across up to four totes.
- Top speed: 1.5 m/s in open aisles; 0.3 m/s within 1 m of a person.
- Battery: 10 hours of continuous operation; 45 minutes to charge from 20% to 80% on the Atlas dock.
- Throughput: up to 400 picks per hour per robot in a tuned site.
- Sensors: 2D LiDAR for safety, 3D LiDAR and stereo cameras for navigation, and a wrist camera for item verification.
- Dimensions: 1.6 m tall, 0.7 m wide, 190 kg.

## Safety

Atlas is designed to ISO 3691-4 with a safety-rated protective stop, a physical emergency stop button on each side, and safety zones that slow and stop the robot before contact. See the safety certification page for site requirements.

## Connectivity

Atlas connects over Wi-Fi 6 to the site's Relay instance and sends a heartbeat every second. If the heartbeat is lost for more than 10 seconds, the robot completes its current motion and stops safely.

## Hardware revisions

Rev C (current) adds the wrist camera and a 20% larger battery. Rev B units can be upgraded in the field. Rev A units are no longer supported by firmware 4.x.
