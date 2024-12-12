import os
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

def load_simulation_profile(power_demand_file):
    power_source_df = pd.read_parquet(power_demand_file)
    power_source_df.sort_values(by="timestamp", inplace=True)
    interval = power_source_df['timestamp'][0]/1000 # seconds
    power_demand = power_source_df['energy_usage'].tolist()
    steps = power_source_df.shape[0]
    energy_supply_csv = pd.read_csv('green_energy_supply.csv')
    production = energy_supply_csv['Production_simulate_half'].tolist()
    if len(production) > steps:
        green_energy_supply = [s * interval for s in production[:steps]]
    else:
        production *= steps // len(production) + 1
        green_energy_supply = [s * interval for s in production[:steps]]
    return steps, interval, power_demand, green_energy_supply

class Battery:
    def __init__(self, capacity, charge_efficiency=0.9, max_charge_rate=10, interval=1):
        self.capacity = capacity * 1000 * 3600 # Battery maximum capacity (Ws)
        self.charge_level = 0  # Current charge level (Wh)
        self.charge_efficiency = charge_efficiency  # Charging efficiency
        self.max_charge_rate = max_charge_rate * 1000 * interval  # Maximum charging rate (Ws)
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
    def __init__(self, green_energy_supply, simulation_steps, interval):
        # Green energy over time (Ws)
        self.green_energy_profile = green_energy_supply
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
    """Smooth energy supplier. """
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

        self.energy_model.green_energy_profile[self.energy_model.current_time % len(self.energy_model.green_energy_profile)] -= clean_energy_used

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


class PowerManagerSingleSupplier:
    """
    Only choose one supplier.
    Use Clean energy first.
    Battery second.
    Non-clean energy third.
    Charge Battery with surplus clean energy.
    Battery could not charge and discharge in the same interval.
    """
    def __init__(self, energy_model, battery):
        self.energy_model = energy_model
        self.battery = battery

    def supply_power(self, demand):
        green_energy = self.energy_model.get_green_energy()
        traditional_energy = self.energy_model.get_traditional_energy()
        self.battery.idle()

        clean_energy_used = 0
        battery_discharge = 0
        non_clean_energy_used = 0

        # Use green energy first
        if green_energy >= demand:
            clean_energy_used = demand
            self.energy_model.green_energy_profile[
                self.energy_model.current_time % len(self.energy_model.green_energy_profile)] -= clean_energy_used

        # Use battery power
        elif self.battery.charge_level >= demand:
            battery_discharge = self.battery.discharge(demand)

        # Use traditional energy
        else:
            non_clean_energy_used = min(traditional_energy, demand)
            demand -= non_clean_energy_used

        return clean_energy_used / 1000, non_clean_energy_used / 1000, battery_discharge / 1000  # Convert back to kilowatts

    def manage_battery(self):
        # Can't charge when discharging.
        if self.battery.state == "discharging":
            self.energy_model.advance_time()
            return

        green_energy = self.energy_model.get_green_energy()
        # Charge the battery if excess green energy is available
        if green_energy > 0 and self.battery.charge_level < self.battery.capacity:
            used_power = self.battery.charge(green_energy)
            self.energy_model.green_energy_profile[self.energy_model.current_time % len(self.energy_model.green_energy_profile)] -= used_power

        # Set battery to idle if fully charged
        if self.battery.charge_level == self.battery.capacity:
            self.battery.idle()

        self.energy_model.advance_time()

class PowerManagerQLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = np.zeros((states, actions))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def discretize_state(self, battery_level, max_battery_level, power_demand, max_power_demand,
                         green_energy_supply, max_green_energy_supply):
        num_battery_intervals = 10
        num_power_demand_intervals = 10
        num_green_energy_intervals = 10

        normalized_battery_level = battery_level / max_battery_level
        normalized_power_demand = power_demand / max_power_demand
        normalized_green_energy_supply = green_energy_supply / max_green_energy_supply

        discrete_battery_level = int(normalized_battery_level * num_battery_intervals)
        discrete_power_demand = int(normalized_power_demand * num_power_demand_intervals)
        discrete_green_energy_supply = int(normalized_green_energy_supply * num_green_energy_intervals)

        state_index = discrete_battery_level * (num_power_demand_intervals * num_green_energy_intervals) + \
                      discrete_power_demand * num_green_energy_intervals + \
                      discrete_green_energy_supply

        return state_index

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q[state]))
        else:
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state):
        self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    def calculate_reward(self, clean_energy_used, non_clean_energy_used, battery_change):
        reward = 0
        reward += 10 * clean_energy_used  # Reward for green energy usage
        reward -= 5 * non_clean_energy_used  # Penalty for traditional energy usage
        reward += 5 * battery_change  # Reward for efficient battery management
        return reward

    def save_q_table(self, file_path):
        np.save(file_path, self.Q)

    def load_q_table(self, file_path):
        if os.path.exists(file_path):
            self.Q = np.load(file_path)
            return True
        return False

class PowerManagerQLearningWrapper:
    def __init__(self, energy_model, battery, q_learning_manager, max_power_demand):
        self.energy_model = energy_model
        self.battery = battery
        self.q_learning_manager = q_learning_manager
        self.max_battery_level = battery.capacity
        self.max_power_demand = max_power_demand
        self.max_green_energy_supply = int(max(energy_model.green_energy_profile) + 1)

    def supply_power(self, demand, current_time = -1):
        if current_time > -1:
            self.energy_model.current_time = current_time
        green_energy = self.energy_model.get_green_energy()
        state = self.q_learning_manager.discretize_state(
            self.battery.charge_level, self.max_battery_level, demand, self.max_power_demand, green_energy,
            self.max_green_energy_supply)

        action = self.q_learning_manager.choose_action(state)

        clean_energy_used, non_clean_energy_used, battery_energy = 0, 0, 0
        if action == 0 and green_energy >= demand:  # Use green energy
            clean_energy_used = demand
            self.energy_model.green_energy_profile[
                self.energy_model.current_time % len(self.energy_model.green_energy_profile)] -= demand

        elif action == 1 and self.battery.charge_level >= demand:  # Use battery
            battery_energy = self.battery.discharge(demand)

        else:  # Use traditional energy
            non_clean_energy_used = demand

        return clean_energy_used / 1000, non_clean_energy_used / 1000, battery_energy / 1000

    def manage_battery(self):
        green_energy = self.energy_model.get_green_energy()
        remain_green_energy = 0
        if self.battery.state != "discharging" and green_energy > 0:
            used_power = self.battery.charge(green_energy)
            remain_green_energy = green_energy - used_power
            self.energy_model.green_energy_profile[
                self.energy_model.current_time % len(self.energy_model.green_energy_profile)] -= used_power

        self.energy_model.advance_time()
        return remain_green_energy

def train_qlearning_manager(q_learning_manager, episodes, analyzer, deep, load_path = None, save_path=None,
                            train_more = False):
    if load_path:
        if q_learning_manager.load_q_table(load_path) and not train_more:
            return
    for episode in tqdm(range(episodes), desc="Training Progress", unit="episode"):
        battery, energy_model = analyzer.generate_models()
        max_power_demand = int(max(analyzer.power_demand) + 1)
        power_manager = PowerManagerQLearningWrapper(energy_model, battery, q_learning_manager, max_power_demand)
        clean_energy_used, non_clean_energy_used, battery_energy = 0, 0, 0
        for i in range(len(analyzer.power_demand)):
            current_time = energy_model.current_time
            battery_train, energy_model_train = analyzer.generate_models()
            for j in range(i, min(i + deep, len(analyzer.power_demand))):
                demand = analyzer.power_demand[j]
                power_manager_train = PowerManagerQLearningWrapper(energy_model_train, battery_train, q_learning_manager, max_power_demand)
                clean_energy, non_clean_energy, battery_energy = power_manager_train.supply_power(demand, current_time)
                power_manager_train.manage_battery()
                current_time = energy_model_train.current_time

                clean_energy_used += clean_energy
                non_clean_energy_used += non_clean_energy
                battery_energy += battery_energy
            demand = analyzer.power_demand[i]
            green_energy = energy_model.get_green_energy()
            state = q_learning_manager.discretize_state(
                power_manager.battery.charge_level, power_manager.max_battery_level, demand,
                power_manager.max_power_demand, green_energy, power_manager.max_green_energy_supply)
            action = q_learning_manager.choose_action(state)

            power_manager.supply_power(demand, current_time)
            green_energy = power_manager.manage_battery()

            demand = 0
            reward = q_learning_manager.calculate_reward(clean_energy_used, non_clean_energy_used, battery_energy)
            next_state = q_learning_manager.discretize_state(
                    power_manager.battery.charge_level, power_manager.max_battery_level, demand,
                    power_manager.max_power_demand, green_energy, power_manager.max_green_energy_supply)
            q_learning_manager.learn(state, action, reward, next_state)
    if save_path:
        q_learning_manager.save_q_table(save_path)

def do_simulation_with_qlearning():
    states = 1000
    actions = 3
    q_learning_manager = PowerManagerQLearning(states, actions)

    power_source_file = "output/simple/raw-output/0/seed=0/powerSource.parquet"

    for i in range(5):
        capacity = 10 + i * 5
        analyzer = EnergyAnalyzer(power_source_file, capacity, None)
        # Training phase
        episodes = 10
        deep = 10
        file_path = f'q_learning_{capacity}_{episodes}_{deep}.csv'
        train_qlearning_manager(q_learning_manager, episodes=episodes, analyzer=analyzer, deep=deep,
                                load_path=file_path, save_path=file_path)
        # Testing phase
        battery, energy_model = analyzer.generate_models()
        max_power_demand = int(max(analyzer.power_demand) + 1)
        power_manager = PowerManagerQLearningWrapper(energy_model, battery, q_learning_manager, max_power_demand)

        total_clean_energy = 0
        total_non_clean_energy = 0
        total_battery_energy = 0

        for demand in analyzer.power_demand:
            clean_energy, non_clean_energy, battery_energy = power_manager.supply_power(demand)
            total_clean_energy += clean_energy
            total_non_clean_energy += non_clean_energy
            total_battery_energy += battery_energy

            power_manager.manage_battery()

        print(f"capacity: {capacity}")
        total = total_clean_energy + total_non_clean_energy + total_battery_energy
        print(f"Total Clean Energy Used: {total_clean_energy / 3600:.2f} kWh, {total_clean_energy / total * 100:.2f}%")
        print(f"Total Non-Clean Energy Used: {total_non_clean_energy / 3600:.2f} kWh, {total_non_clean_energy / total * 100:.2f}%")
        print(f"Total Battery Energy Used: {total_battery_energy / 3600:.2f} kWh, {total_battery_energy / total * 100:.2f}%")
        print(f"Total Energy Used: {total / 3600:.2f} kWh")
        print('-' * 50)


class EnergyAnalyzer:
    def __init__(self, power_demand_file, battery_capacity, power_manager):
        self.power_demand_file = power_demand_file
        self.tasks = []
        self.power_demand = []
        self.power_manager = power_manager
        steps, interval, power_demand, green_energy_supply = load_simulation_profile(power_demand_file)
        self.interval = interval
        self.steps = steps
        self.power_demand = power_demand
        self.green_energy_supply = green_energy_supply
        self.battery_capacity = battery_capacity

    def generate_models(self):
        energy_model = EnergyModel(self.green_energy_supply.copy(), self.steps, self.interval)
        battery = Battery(capacity=self.battery_capacity, interval=self.interval)
        return battery, energy_model

    def simulate(self):
        """Simulate power usage and calculate clean energy usage."""
        battery, energy_model = self.generate_models()
        power_manager = self.power_manager(energy_model, battery)

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

def do_simulation(power_manager):
    print(f'Run simulation with {power_manager}')
    power_source_file = "output/simple/raw-output/0/seed=0/powerSource.parquet"

    for i in range(5):
        capacity=10 + i * 5

        analyzer = EnergyAnalyzer(power_source_file, capacity, power_manager)
        results = analyzer.simulate()

        print(f"capacity: {capacity}")
        total = results['total_clean_energy'] + results['total_non_clean_energy'] + results['total_battery_energy']
        print(f"Total Clean Energy Used: {results['total_clean_energy']/3600:.2f} kWh, {results['total_clean_energy'] / total * 100:.2f}%")
        print(f"Total Non-Clean Energy Used: {results['total_non_clean_energy']/3600:.2f} kWh, {results['total_non_clean_energy'] / total * 100:.2f}%")
        print(f"Total Battery Energy Used: {results['total_battery_energy']/3600:.2f} kWh, {results['total_battery_energy'] / total * 100:.2f}%")
        print(f"Total Energy Used: {total/3600:.2f} kWh")
        print('-' * 50)

# Example usage
if __name__ == "__main__":
    # do_simulation(PowerManager)
    # do_simulation(PowerManagerSingleSupplier)
    do_simulation_with_qlearning()
