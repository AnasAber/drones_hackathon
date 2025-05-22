# enhanced_drone_system.py

import folium
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum

global total_time
total_time = 0

# ----- Constants and Configuration -----

class DroneModel(Enum):
    CW_30E = "JOUAV_CW_30E"

@dataclass
class DroneSpecs:
    model: DroneModel
    max_flight_time_minutes: int = 480  # 8 hours
    cruise_speed_kmh: int = 90  # Estimated cruise speed
    mission_time_minutes: int = 5  # Time to complete EXAMINE mission
    cost_usd: int = 30000

# ----- Enhanced Data Classes -----

class GeoCalculator:
    """Handles all geospatial calculations"""
    EARTH_RADIUS_KM = 6371.0
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers using Haversine formula"""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return GeoCalculator.EARTH_RADIUS_KM * c
    
    @staticmethod
    def calculate_flight_time(distance_km: float, speed_kmh: float) -> float:
        """Calculate flight time in minutes"""
        return (distance_km / speed_kmh) * 60

class Drone:
    def __init__(self, drone_id: str, base: 'DroneBase', specs: DroneSpecs = DroneSpecs(DroneModel.CW_30E)):
        self.drone_id = drone_id
        self.base = base
        self.specs = specs
        self.status = "available"  # available, in_mission, charging, maintenance
        self.assigned_tickets: List['Ticket'] = []
        self.current_location = (base.latitude, base.longitude)
        self.remaining_flight_time = specs.max_flight_time_minutes
        self.mission_start_time: Optional[datetime] = None
        self.total_distance_flown = 0.0
        self.missions_completed = 0
        
    def can_accept_ticket(self, ticket: 'Ticket') -> Tuple[bool, str]:
        """Check if drone can accept a ticket based on flight time constraints"""
        if self.status != "available":
            return False, f"Drone not available (status: {self.status})"
            
        # Calculate round trip distance
        distance_to_target = GeoCalculator.calculate_distance(
            self.base.latitude, self.base.longitude,
            ticket.latitude, ticket.longitude
        )
        round_trip_distance = distance_to_target * 2
        
        # Calculate required flight time
        flight_time = GeoCalculator.calculate_flight_time(
            round_trip_distance, self.specs.cruise_speed_kmh
        )
        mission_time = self.specs.mission_time_minutes
        total_time_required = flight_time + mission_time
        
        if total_time_required > self.specs.max_flight_time_minutes:
            return False, f"Mission requires {total_time_required:.1f} min, exceeds max flight time"
            
        return True, "Can accept mission"
    
    """For now, implement single ticket assignment"""
    def assign_tickets(self, tickets: List['Ticket']) -> bool:
        if not tickets:
            return False

        ticket = tickets[0]
        can_accept, reason = self.can_accept_ticket(ticket)
        
        if can_accept:
            self.status = "in_mission"
            self.assigned_tickets = [ticket]
            self.mission_start_time = datetime.now()
            ticket.assign(self)
            return True
        
        return False
    
    def complete_mission(self) -> Dict:
        """complete current mission and return statistics"""
        if not self.assigned_tickets:
            return {}
            
        ticket = self.assigned_tickets[0]
        distance_flown = GeoCalculator.calculate_distance(
            self.base.latitude, self.base.longitude,
            ticket.latitude, ticket.longitude
        ) * 2  # Round trip
        
        flight_time = GeoCalculator.calculate_flight_time(
            distance_flown, self.specs.cruise_speed_kmh
        )
        
        mission_stats = {
            "drone_id": self.drone_id,
            "ticket_id": ticket.sample_id,
            "distance_km": distance_flown,
            "flight_time_minutes": flight_time + self.specs.mission_time_minutes,
            "mission_start": self.mission_start_time,
            "mission_end": self.mission_start_time + flight_time + self.specs.mission_time_minutes # datetime.now()
        }

        total_time += mission_stats["mission_end"] - mission_stats["mission_end"]
    
        # Update drone state
        self.status = "available"
        self.total_distance_flown += distance_flown
        self.missions_completed += 1
        self.assigned_tickets = []
        self.mission_start_time = None
        
        return mission_stats

class DroneBase:
    def __init__(self, city_name: str, latitude: float, longitude: float):
        self.city_name = city_name
        self.latitude = latitude
        self.longitude = longitude
        self.drones: List[Drone] = []
        self.operational_hours = (8, 17)  # 9AM to 5PM
        self.tickets_completed = 0
        self.total_revenue = 0.0
        
    def add_drones(self, count: int):
        """Add specified number of drones to base"""
        for i in range(count):
            drone_id = f"{self.city_name[:3].upper()}-{len(self.drones)+1:03d}"
            drone = Drone(drone_id, self)
            self.drones.append(drone)
    
    def get_available_drones(self) -> List[Drone]:
        """Get all available drones during operational hours"""
        current_hour = datetime.now().hour
        if not (self.operational_hours[0] <= current_hour <= self.operational_hours[1]):
            return []  # Base out of work hours
            
        return [d for d in self.drones if d.status == "available"]
    
    def get_best_drone_for_ticket(self, ticket: 'Ticket') -> Optional[Drone]:
        """Get the best available drone for a ticket"""
        available_drones = self.get_available_drones()
        suitable_drones = []
        
        for drone in available_drones:
            can_accept, reason = drone.can_accept_ticket(ticket)
            if can_accept:
                distance = GeoCalculator.calculate_distance(
                    self.latitude, self.longitude,
                    ticket.latitude, ticket.longitude
                )
                suitable_drones.append((drone, distance))
        
        if suitable_drones:
            # Return drone with shortest distance to target
            return min(suitable_drones, key=lambda x: x[1])[0]
        return None

class Ticket:
    def __init__(self, sample_id: str, latitude: float, longitude: float, 
                 ndvi: float):
        self.sample_id = sample_id
        self.latitude = latitude
        self.longitude = longitude
        self.ndvi = ndvi
        self.assigned = False
        self.assigned_drone: Optional[Drone] = None
        self.priority = self._calculate_priority()
        self.created_time = datetime.now()
        self.assigned_time: Optional[datetime] = None
        self.completed_time: Optional[datetime] = None
        
    def _calculate_priority(self) -> int:
        """Calculate ticket priority based on NDVI (lower NDVI => higher priority)"""
        if self.ndvi < 0.3:
            return 1  # Critical
        elif self.ndvi < 0.5:
            return 2  # High
        else:
            return 3  # Medium
    
    def assign(self, drone: Drone):
        """Assign ticket to drone"""
        self.assigned = True
        self.assigned_drone = drone
        self.assigned_time = datetime.now()

# ----- Assignment Algorithms -----

class AssignmentStrategy(Enum):
    
    # these are three strategies that we'll implement
    NEAREST_BASE = "nearest_base" 
    ECONOMIC_OPTIMAL = "economic_optimal"
    PRIORITY_BASED = "priority_based"

class TicketAssignmentSystem:
    def __init__(self, strategy: AssignmentStrategy = AssignmentStrategy.NEAREST_BASE):
        self.strategy = strategy
        self.assignment_history = []
        
    def assign_tickets(self, bases: List[DroneBase], tickets: List[Ticket]) -> Dict:
        """Assign tickets to drones based on selected strategy"""
        print("entered assign_tickets....")
        assignments = {"successful": 0, "failed": 0, "details": []}
        
        # Sort tickets by priority (critical first)
        unassigned_tickets = [t for t in tickets if not t.assigned]
        unassigned_tickets.sort(key=lambda x: x.priority) # sort by priority (NDVI)
        if unassigned_tickets:
            print("unassigned tickets is full...")
        for ticket in unassigned_tickets:
            if self.strategy == AssignmentStrategy.NEAREST_BASE:
                success = self._assign_nearest_base(ticket, bases)
            elif self.strategy == AssignmentStrategy.ECONOMIC_OPTIMAL:
                success = self._assign_economic_optimal(ticket, bases)
            else:  # PRIORITY_BASED
                success = self._assign_priority_based(ticket, bases)
                
            if success:
                assignments["successful"] += 1
                assignments["details"].append({
                    "ticket_id": ticket.ndvi+uuid.uuid3(ticket.sample_id),
                    "assigned_to": ticket.assigned_drone.drone_id,
                    "base": ticket.assigned_drone.base.city_name
                })
            else:
                assignments["failed"] += 1
        
        return assignments
    
    def _assign_nearest_base(self, ticket: Ticket, bases: List[DroneBase]) -> bool:
        """Assign to the nearest base with available drone"""
        base_distances = []
        for base in bases:
            distance = GeoCalculator.calculate_distance(
                base.latitude, base.longitude,
                ticket.latitude, ticket.longitude
            )

            drone = base.get_best_drone_for_ticket(ticket)

            if drone:
                base_distances.append((base, drone, distance))
        
        if base_distances:
            minim = min(base_distances, key=lambda x: x[2])
            _, best_drone, _ = minim
            return best_drone.assign_tickets([ticket])
        return False
    
    def _assign_economic_optimal(self, ticket: Ticket, bases: List[DroneBase]) -> bool:
        """Assign based on economic efficiency (minimize cost per mission)"""
        best_option = None
        best_efficiency = float('inf')
        
        for base in bases:
            drone = base.get_best_drone_for_ticket(ticket)
            if drone:
                distance = GeoCalculator.calculate_distance(
                    base.latitude, base.longitude,
                    ticket.latitude, ticket.longitude
                ) * 2  # Round trip
                
                flight_time = GeoCalculator.calculate_flight_time(
                    distance, drone.specs.cruise_speed_kmh
                )
                total_time = flight_time + drone.specs.mission_time_minutes
                
                # Simple efficiency metric: time per mission
                efficiency = total_time / 1  # Could be adjusted for revenue per ticket
                
                if efficiency < best_efficiency:
                    best_efficiency = efficiency
                    best_option = drone
        
        if best_option:
            print(f"Best drone is getting assigned to a ticket rn in _assign_economic_optimal() function")
            return best_option.assign_tickets([ticket])
        return False
    
    def _assign_priority_based(self, ticket: Ticket, bases: List[DroneBase]) -> bool:
        """Assign based on ticket priority and base capacity"""
        # for high priority tickets, begin with bases with more available drones
        if ticket.priority == 1:  # Critical
            bases_by_capacity = sorted(bases, 
                                     key=lambda b: len(b.get_available_drones()), 
                                     reverse=True)
        else:
            bases_by_capacity = bases
            
        for base in bases_by_capacity:
            drone = base.get_best_drone_for_ticket(ticket)
            if drone:
                return drone.assign_tickets([ticket])
        return False

# ----- Data Loading Functions -----

def load_drone_bases(filepath: str) -> List[DroneBase]:
    """Load drone bases from CSV file"""
    try:
        # Try to read the CSV file
        df = pd.read_csv(filepath)
        
        # Handle different possible column names
        city_col = None
        lat_col = None
        lng_col = None
        
        for col in df.columns:
            col_upper = col.upper()
            if 'CITY' in col_upper or 'NAME' in col_upper:
                city_col = col
            elif 'LAT' in col_upper and 'LNG' not in col_upper:
                lat_col = col
            elif 'LNG' in col_upper or 'LON' in col_upper:
                lng_col = col
        
        if not all([city_col, lat_col, lng_col]):
            raise ValueError("Could not identify required columns in drone_bases.csv")
        
        bases = []
        for _, row in df.iterrows():
            base = DroneBase(
                city_name=str(row[city_col]),
                latitude=float(row[lat_col]),
                longitude=float(row[lng_col])
            )
            bases.append(base)
        
        print(f"✅ Loaded {len(bases)} drone bases from {filepath}")
        return bases
        
    except FileNotFoundError:
        print(f"❌ Could not find {filepath}, using default bases")
        return create_default_bases()
    except Exception as e:
        print(f"❌ Error loading drone bases: {e}, using default bases")
        return create_default_bases()

def load_crop_targets(filepath: str, ndvi_threshold: float = 0.5) -> List[Ticket]:
    """Load crop targets from CSV file"""
    try:
        df = pd.read_csv(filepath)
        
        # Handle different possible column names
        sample_col = None
        lat_col = None
        lng_col = None
        ndvi_col = None
        crop_col = None
        
        for col in df.columns:
            col_upper = col.upper()
            if 'SAMPLE' in col_upper or 'ID' in col_upper:
                sample_col = col
            elif 'LAT' in col_upper and 'LNG' not in col_upper:
                lat_col = col
            elif 'LNG' in col_upper or 'LON' in col_upper:
                lng_col = col
            elif 'NDVI' in col_upper:
                ndvi_col = col
            elif 'CROP' in col_upper:
                crop_col = col
        
        if not all([sample_col, lat_col, lng_col, ndvi_col]):
            raise ValueError("Could not identify required columns in crop_targets.csv")
        
        # Filter by NDVI threshold
        df_filtered = df[df[ndvi_col] < ndvi_threshold]
        
        tickets = []
        for _, row in df_filtered.iterrows():
            ticket = Ticket(
                sample_id=str(row[sample_col]),
                latitude=float(row[lat_col]),
                longitude=float(row[lng_col]),
                ndvi=float(row[ndvi_col]),
            )
            tickets.append(ticket)
        
        print(f"✅ Loaded {len(tickets)} crop targets from {filepath} (NDVI < {ndvi_threshold})")
        return tickets
        
    except FileNotFoundError:
        print(f"❌ Could not find {filepath}, using generated tickets")
        return None
    except Exception as e:
        print(f"❌ Error loading crop targets: {e}, using generated tickets")
        return e


# ----- Analytics System -----

class AnalyticsEngine:
    def __init__(self):
        self.simulation_data = []
        self.mission_history = []

    def analyze_scenario(self, bases: List[DroneBase], tickets: List[Ticket], 
                        scenario_name: str) -> Dict:
        """Analyze a specific scenario configuration"""
        total_drones = sum(len(base.drones) for base in bases)
        total_tickets = len([t for t in tickets if not t.assigned])
        assigned_tickets = len([t for t in tickets if t.assigned])

        # Calculate theoretical completion time
        avg_mission_time = total_time # minutes (flight + mission time)
        theoretical_min_time = (total_tickets * avg_mission_time) / total_drones
        
        # Calculate coverage efficiency
        coverage_efficiency = assigned_tickets / len(tickets) * 100
        
        # Calculate economic metrics
        total_drone_cost = total_drones * 30000  # $30k per drone
        operational_cost_per_hour = total_drones * 20  # let's say 20$ per drone, including fuel and stuff
        
        analysis = {
            "scenario_name": scenario_name,
            "total_bases": len(bases),
            "total_drones": total_drones,
            "drones_per_base": total_drones / len(bases),
            "total_tickets": len(tickets),
            "assigned_tickets": assigned_tickets,
            "coverage_efficiency_percent": coverage_efficiency,
            "theoretical_completion_hours": theoretical_min_time / 60,
            "total_drone_investment_usd": total_drone_cost,
            "estimated_daily_operational_cost": operational_cost_per_hour * 8,  # 8 hours in a day
            "base_utilization": self._calculate_base_usage(bases)
        }
        return analysis
    
    def _calculate_base_usage(self, bases: List[DroneBase]) -> Dict[str, float]:
        """calculate usage rate for each base"""
        usage = {}
        for base in bases:
            busy_drones = len([d for d in base.drones if d.status != "available"])
            total_drones = len(base.drones)
            usage[base.city_name] = (busy_drones / total_drones * 100) if total_drones > 0 else 0
        return usage
    
    def compare_scenarios(self, scenarios: List[Dict]) -> Dict:
        """compare multiple scenarios and provide recommendations"""
        comparison = {
            "scenarios": scenarios,
            "best_coverage": max(scenarios, key=lambda x: x["coverage_efficiency_percent"]),
            "most_efficient": min(scenarios, key=lambda x: x["theoretical_completion_hours"]),
            "most_economical": min(scenarios, key=lambda x: x["total_drone_investment_usd"]),
            "recommendations": []
        }
        
        # Generate recommendations
        if comparison["best_coverage"]["coverage_efficiency_percent"] < 80:
            comparison["recommendations"].append("Consider increasing drone fleet size for better coverage")
        
        if comparison["most_efficient"]["theoretical_completion_hours"] > 24:
            comparison["recommendations"].append("Mission completion may take more than a day - consider optimization")
        
        return comparison

# ----- Visualization Enhancements -----

def create_enhanced_visualization(bases: List[DroneBase], tickets: List[Ticket], 
                                analytics: Dict, filename: str = "enhanced_drone_map.html"):
    """Create enhanced visualization with analytics overlay"""
    philippines_map = folium.Map(location=[12.8797, 121.7740], zoom_start=6)
    
    # Add base markers with drone count info
    for base in bases:
        available_drones = len(base.get_available_drones())
        busy_drones = len([d for d in base.drones if d.status == "in_mission"])
        
        popup_text = f"""
        <b>{base.city_name} Base</b><br>
        Total Drones: {len(base.drones)}<br>
        Available: {available_drones}<br>
        In Mission: {busy_drones}<br>
        Missions Completed: {base.tickets_completed}
        """
    
        folium.Marker(
            location=[base.latitude, base.longitude],
            popup=folium.Popup(popup_text, max_width=200),
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(philippines_map)
    
    # Add ticket markers with priority color coding
    priority_colors = {1: "red", 2: "orange", 3: "yellow"}
    
    for ticket in tickets:
        color = "green" if ticket.assigned else priority_colors.get(ticket.priority, "gray")
        
        popup_text = f"""
        <b>Ticket {ticket.sample_id}</b><br>
        NDVI: {ticket.ndvi:.3f}<br>
        Priority: {ticket.priority}<br>
        Status: {'Assigned' if ticket.assigned else 'Unassigned'}
        """
        
        if ticket.assigned:
            popup_text += f"<br>Assigned to: {ticket.assigned_drone.drone_id}"
        
        folium.Marker(
            location=[ticket.latitude, ticket.longitude],
            popup=folium.Popup(popup_text, max_width=200),
            icon=folium.Icon(color=color, icon="leaf")
        ).add_to(philippines_map)
        
        # Add flight path for assigned tickets
        if ticket.assigned and ticket.assigned_drone:
            drone = ticket.assigned_drone
            base_lat, base_lon = drone.base.latitude, drone.base.longitude
            folium.PolyLine(
                locations=[(base_lat, base_lon), (ticket.latitude, ticket.longitude)],
                color="purple",
                weight=2,
                opacity=0.7,
                popup=f"{drone.drone_id} → {ticket.sample_id}"
            ).add_to(philippines_map)
    
    # Add analytics info box
    analytics_html = f"""
    <div style="position: fixed; 
                top: 10px; left: 10px; width: 300px; height: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <h4>Mission Analytics</h4>
    <p><b>Total Bases:</b> {analytics['total_bases']}</p>
    <p><b>Total Drones:</b> {analytics['total_drones']}</p>
    <p><b>Coverage Efficiency:</b> {analytics['coverage_efficiency_percent']:.1f}%</p>
    <p><b>Est. Completion:</b> {analytics['theoretical_completion_hours']:.1f} hours</p>
    <p><b>Investment Cost:</b> ${analytics['total_drone_investment_usd']:,}</p>
    </div>
    """
    philippines_map.get_root().add_child(folium.Element(analytics_html))
    
    philippines_map.save(filename)
    print(f"📊 Enhanced map saved: {filename}")

# ----- Main Simulation Function -----

def run_comprehensive_simulation():
    """Run comprehensive simulation with all enhancements"""
    print("Starting Enhanced Drone Swarm Logistics Simulation...")
    print("=" * 60)
    
    print("\n Loading Data...")
    bases = load_drone_bases("../data/drone_bases_test.csv")
    tickets = load_crop_targets("../data/crop_data_test.csv", ndvi_threshold=0.5)
    
    analytics = AnalyticsEngine()
    
    scenarios_to_test = [
        {"name": "Scenario 1: Balanced", "drones_per_base": 10},
        {"name": "Scenario 2: Few bases, many drones", "bases_to_use": 5, "drones_per_base": 100},
        {"name": "Scenario 3: Many bases, few drones", "drones_per_base": 3},
    ]

    scenario_results = []
    
    for scenario_config in scenarios_to_test:
        print(f"\n Testing {scenario_config['name']}")
        print("-" * 40)
        
        # Setup scenario
        if "bases_to_use" in scenario_config:
            # Use only first N bases
            scenario_bases = bases[:scenario_config["bases_to_use"]]
        else:
            scenario_bases = bases.copy()
        
        # Add drones to bases
        for base in scenario_bases:
            base.drones = []  # Reset drones
            base.add_drones(scenario_config["drones_per_base"])
        
        # Reset ticket assignments
        for ticket in tickets:
            ticket.assigned = False
            ticket.assigned_drone = None
        
        # Run assignment
        assignment_system = TicketAssignmentSystem(AssignmentStrategy.PRIORITY_BASED)
        print(f"We got assignment system running {assignment_system}")
        assignment_results = assignment_system.assign_tickets(scenario_bases, tickets)
        
        # Analyze results
        scenario_analysis = analytics.analyze_scenario(scenario_bases, tickets, scenario_config["name"])
        scenario_results.append(scenario_analysis)
        
        # Print results
        print(f"✅ Assigned: {assignment_results['successful']} tickets")
        print(f"❌ Failed: {assignment_results['failed']} tickets")
        print(f"📊 Coverage: {scenario_analysis['coverage_efficiency_percent']:.1f}%")
        print(f"⏱ Est. completion: {scenario_analysis['theoretical_completion_hours']:.1f} hours")
        
        # Create visualization
        filename = f"scenario_{scenario_config['name'].split()[1].lower()}_map.html"
        create_enhanced_visualization(scenario_bases, tickets, scenario_analysis, filename)
    
    # Compare scenarios
    print(f"\n📈 Scenario Comparison")
    print("=" * 40)
    comparison = analytics.compare_scenarios(scenario_results)
    
    print(f"🏆 Best Coverage: {comparison['best_coverage']['scenario_name']} "
          f"({comparison['best_coverage']['coverage_efficiency_percent']:.1f}%)")
    print(f"⚡ Most Efficient: {comparison['most_efficient']['scenario_name']} "
          f"({comparison['most_efficient']['theoretical_completion_hours']:.1f} hours)")
    print(f"💰 Most Economical: {comparison['most_economical']['scenario_name']} "
          f"(${comparison['most_economical']['total_drone_investment_usd']:,})")
    
    print(f"\n💡 Recommendations:")
    for rec in comparison['recommendations']:
        print(f"   • {rec}")
    
    # Export comprehensive results
    export_comprehensive_results(scenario_results, comparison, tickets)
    
    print(f"\n✅ Comprehensive simulation complete!")
    return scenario_results, comparison

def export_comprehensive_results(scenario_results: List[Dict], comparison: Dict, tickets: List[Ticket]):
    """Export all results to CSV files"""
    # Scenario comparison
    df_scenarios = pd.DataFrame(scenario_results)
    df_scenarios.to_csv("scenario_analysis.csv", index=False)
    
    # Detailed ticket analysis
    ticket_data = []
    for ticket in tickets:
        ticket_data.append({
            "sample_id": ticket.sample_id,
            "latitude": ticket.latitude,
            "longitude": ticket.longitude,
            "ndvi": ticket.ndvi,
            "priority": ticket.priority,
            "assigned": ticket.assigned,
            "assigned_drone": ticket.assigned_drone.drone_id if ticket.assigned_drone else None,
            "assigned_base": ticket.assigned_drone.base.city_name if ticket.assigned_drone else None
        })
    
    df_tickets = pd.DataFrame(ticket_data)
    df_tickets.to_csv("detailed_ticket_analysis.csv", index=False)
    
    # Export comparison as JSON for detailed analysis
    with open("scenario_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print("📁 Results exported: scenario_analysis.csv, detailed_ticket_analysis.csv, scenario_comparison.json")

if __name__ == "__main__":
    # Ensure data_logs directory exists
    import os
    os.makedirs("data_logs", exist_ok=True)
    
    run_comprehensive_simulation()