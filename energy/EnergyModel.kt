package org.opendc.simulator.compute.energy

public class EnergyModel(
    private val greenEnergyProfile: List<Double>,   // Green energy supply over time (Ws)
    simulationSteps: Int
) {
    private val traditionalEnergyProfile: List<Double> = List(simulationSteps) { 9999.0 }
    private var currentTime: Int = 0
        private set

    public fun getGreenEnergy(): Double {
        return greenEnergyProfile[currentTime % greenEnergyProfile.size]
    }

    public fun getTraditionalEnergy(): Double {
        return traditionalEnergyProfile[currentTime % traditionalEnergyProfile.size]
    }

    public fun advanceTime() {
        currentTime += 1
    }
}
