# Drone Swarm Logistics Simulation


### Project Overview

This project simulates the operation of a drone swarm logistics system designed for agricultural monitoring and crop health analysis. It focuses on optimizing drone deployment from multiple bases to efficiently cover crop monitoring tasks spread across a region.

The simulation tests various operational scenarios, evaluates drone assignment strategies, and provides detailed analytics to help improve mission success rates, resource utilization, and overall system efficiency.

### Key Features

Scenario-Based Testing: Explore different configurations such as varying the number of drone bases and the number of drones per base to find the most effective setup.

Dynamic Ticket Assignment: Assign crop monitoring tasks (tickets) to drones based on proximity and drone availability, simulating real-world operational constraints.

Comprehensive Analytics: Analyze key metrics including coverage efficiency, mission completion time estimates, drone utilization, and investment costs for each scenario.

Interactive Visualizations: Generate detailed, interactive maps displaying drone bases, ticket locations with priority statuses, flight paths, and real-time analytics overlays.

Multi-Day Simulation: Simulate multiple days of operations where unassigned tasks from previous days carry over, allowing analysis of long-term operational capacity and backlog clearance.

Data Export: Export detailed scenario results and ticket assignment data for further analysis and reporting.

### Purpose and Benefits

This simulation tool is designed to assist agricultural planners and drone operations managers by:

 1- Evaluating different drone deployment strategies under realistic conditions.

 2- Identifying bottlenecks and limitations in drone availability and ticket coverage.

 3- Providing data-driven insights to optimize drone fleet size and base locations.

 4- Supporting informed decision-making to improve monitoring efficiency and reduce operational costs.

## How It Works

1-Loading Data: Drone bases and crop monitoring tickets are loaded from prepared datasets, representing real-world geographic locations and crop health indicators.

2-Configuring Scenarios: Users define scenarios by specifying parameters such as the number of drones per base or limiting the number of active bases.

3-Running Simulations: For each scenario, the system assigns tickets to drones, calculates analytics, and produces interactive maps showing the operational status and coverage.

4-Multi-Day Operations: Unassigned tickets from a simulation day are carried forward to the next day, simulating continuous operations until all tickets are addressed or no further progress can be made.

Results and Reporting: Detailed reports on drone utilization, task assignments, and scenario comparisons are generated and saved for review.

### Intended Users

 * Agricultural planners and researchers focused on precision farming.

 * Drone fleet managers and logistics coordinators.

 * Developers and data scientists working on drone operations optimization.

 * Anyone interested in modeling and analyzing drone-based delivery or monitoring systems.

### Requirements
 * A Python environment with standard data analysis and visualization libraries.

 * Data inputs including drone base locations and crop target information.

 * Basic understanding of drone operations and logistics simulation concepts.

