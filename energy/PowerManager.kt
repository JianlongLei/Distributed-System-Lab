package org.opendc.simulator.compute.energy

public class PowerManager(
    private val energyModel: EnergyModel,
    private val battery: Battery
) {
    public fun supplyPower(demand: Double): Triple<Double, Double, Double> {
        var remainingDemand = demand
        val greenEnergy = energyModel.getGreenEnergy()
        val traditionalEnergy = energyModel.getTraditionalEnergy()
        var cleanEnergyUsed = 0.0
        var batteryDischarge = 0.0
        var nonCleanEnergyUsed = 0.0

        // Use green energy first
        if (greenEnergy >= remainingDemand) {
            cleanEnergyUsed = remainingDemand
            remainingDemand = 0.0
        } else {
            cleanEnergyUsed = greenEnergy
            remainingDemand -= greenEnergy
        }

        // Use battery power
        if (remainingDemand > 0 && battery.chargeLevel > 0) {
            batteryDischarge = battery.discharge(remainingDemand)
            remainingDemand -= batteryDischarge
        }

        // Use traditional energy
        if (remainingDemand > 0) {
            nonCleanEnergyUsed = minOf(traditionalEnergy, remainingDemand)
            remainingDemand -= nonCleanEnergyUsed
        }

        energyModel.advanceTime()

        return Triple(cleanEnergyUsed / 1000, nonCleanEnergyUsed / 1000, batteryDischarge / 1000) // Convert to kWh
    }

    public fun manageBattery() {
        val greenEnergy = energyModel.getGreenEnergy()

        // Charge the battery if excess green energy is available
        if (greenEnergy > 0 && battery.chargeLevel < battery.capacity) {
            val usedPower = battery.charge(greenEnergy)
        }

        // Set battery to idle if fully charged
        if (battery.chargeLevel == battery.capacity) {
            battery.idle()
        }

        energyModel.advanceTime()
    }
}
