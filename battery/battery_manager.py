import random

import pandas as pd

class Battery:
    def __init__(self, capacity, charge_efficiency=0.9, max_charge_rate=10):
        self.capacity = capacity * 1000 * 3600 # Battery maximum capacity (Ws)
        self.charge_level = 0  # Current charge level (Wh)
        self.charge_efficiency = charge_efficiency  # Charging efficiency
        self.max_charge_rate = max_charge_rate * 1000  # Maximum charging rate (W)
        self.state = "idle"  # State: charging, discharging, idle

    def charge(self, power):
        self.state = "charging"
        charge_power = min(power, self.max_charge_rate)  # Limit charging rate
        charge_amount = charge_power * self.charge_efficiency
        self.charge_level = min(self.capacity, self.charge_level + charge_amount)
        return charge_power  # Actual energy consumed for charging

    def discharge(self, demand):
        self.state = "discharging"
        discharge_amount = min(self.charge_level, demand)
        self.charge_level -= discharge_amount
        return discharge_amount  # Energy supplied to the load

    def idle(self):
        self.state = "idle"

class EnergyModel:
    def __init__(self, simulation_steps, interval):
        # Green energy over time (Ws)
        self.green_energy_profile = [random.uniform(0, 400) * interval for _ in range(simulation_steps)]
        # Traditional energy over time (Ws)
        self.traditional_energy_profile = [9999 * interval for _ in range(simulation_steps)]
        self.current_time = 0

    def get_green_energy(self):
        return self.green_energy_profile[self.current_time % len(self.green_energy_profile)]

    def get_traditional_energy(self):
        return self.traditional_energy_profile[self.current_time % len(self.traditional_energy_profile)]

    def advance_time(self):
        self.current_time += 1

class PowerManager:
    def __init__(self, energy_model, battery):
        self.energy_model = energy_model
        self.battery = battery

    def supply_power(self, demand):
        green_energy = self.energy_model.get_green_energy()
        traditional_energy = self.energy_model.get_traditional_energy()
        power_supplied = 0

        # Use green energy first
        clean_energy_used = 0
        if green_energy >= demand:
            clean_energy_used = demand
            power_supplied += demand
            demand = 0
        else:
            clean_energy_used = green_energy
            power_supplied += green_energy
            demand -= green_energy

        # Use battery power
        battery_discharge = 0
        if demand > 0 and self.battery.charge_level > 0:
            battery_discharge = self.battery.discharge(demand)
            power_supplied += battery_discharge
            demand -= battery_discharge

        # Use traditional energy
        non_clean_energy_used = 0
        if demand > 0:
            non_clean_energy_used = min(traditional_energy, demand)
            power_supplied += non_clean_energy_used
            demand -= non_clean_energy_used

        return clean_energy_used / 1000, non_clean_energy_used / 1000, battery_discharge / 1000  # Convert back to kilowatts

    def manage_battery(self):
        green_energy = self.energy_model.get_green_energy()
        # Charge the battery if excess green energy is available
        if green_energy > 0 and self.battery.charge_level < self.battery.capacity:
            used_power = self.battery.charge(green_energy)
            self.energy_model.green_energy_profile[self.energy_model.current_time % len(self.energy_model.green_energy_profile)] -= used_power

        # Set battery to idle if fully charged
        if self.battery.charge_level == self.battery.capacity:
            self.battery.idle()

        self.energy_model.advance_time()

class EnergyAnalyzer:
    def __init__(self, power_demand_file):
        self.power_demand_file = power_demand_file
        self.tasks = []
        self.power_demand = []
        self.interval = 0
        self.steps = 0

    def load_output(self):
        """Load the task and power source data from Parquet files."""
        power_source_df = pd.read_parquet(self.power_demand_file)
        power_source_df.sort_values(by="timestamp", inplace=True)
        self.interval = power_source_df['timestamp'][0]/1000 # seconds
        self.power_demand = power_source_df['energy_usage'].tolist()
        self.steps = power_source_df.shape[0]

    def simulate(self, battery_capacity):
        """Simulate power usage and calculate clean energy usage."""
        energy_model = EnergyModel(self.steps, self.interval)
        battery = Battery(capacity=battery_capacity)
        power_manager = PowerManager(energy_model, battery)

        total_clean_energy = 0
        total_non_clean_energy = 0
        total_battery_energy = 0

        for demand in self.power_demand:
            clean_energy, non_clean_energy, battery_energy = power_manager.supply_power(demand)
            total_clean_energy += clean_energy
            total_non_clean_energy += non_clean_energy
            total_battery_energy += battery_energy

            power_manager.manage_battery()

        return {
            "total_clean_energy": total_clean_energy,
            "total_non_clean_energy": total_non_clean_energy,
            "total_battery_energy": total_battery_energy
        }

# Example usage
if __name__ == "__main__":
    task_file = "output/tasks.parquet"
    power_source_file = "output/simple/raw-output/0/seed=0/powerSource.parquet"

    for i in range(5):
        analyzer = EnergyAnalyzer(power_source_file)
        analyzer.load_output()

        capacity=40 + i * 5
        results = analyzer.simulate(battery_capacity=capacity)
        print(f"capacity: {capacity}")
        total = results['total_clean_energy'] + results['total_non_clean_energy'] + results['total_battery_energy']
        print(f"Total Clean Energy Used: {results['total_clean_energy']:.2f} kWs, {results['total_clean_energy'] / total * 100:.2f}%")
        print(f"Total Non-Clean Energy Used: {results['total_non_clean_energy']:.2f} kWs, {results['total_non_clean_energy'] / total * 100:.2f}%")
        print(f"Total Battery Energy Used: {results['total_battery_energy']:.2f} kWs, {results['total_battery_energy'] / total * 100:.2f}%")
        print(f"Total Energy Used: {total:.2f} kWs")
        print('-' * 50)
