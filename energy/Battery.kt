package org.opendc.simulator.compute.energy

public class Battery(
    public val capacity: Double,        // Maximum battery capacity in Ws
    private val chargeEfficiency: Double = 0.9,
    private val maxChargeRate: Double = 10.0 * 1000
) {
    public var chargeLevel: Double = 0.0        // Current charge level in Ws
    public var state: String = "idle"           // State: charging, discharging, idle

    public fun charge(power: Double): Double {
        state = "charging"
        val chargePower = minOf(power, maxChargeRate)
        val chargeAmount = chargePower * chargeEfficiency
        chargeLevel = minOf(capacity, chargeLevel + chargeAmount)
        return chargePower
    }

    public fun discharge(demand: Double): Double {
        state = "discharging"
        val dischargeAmount = minOf(chargeLevel, demand)
        chargeLevel -= dischargeAmount
        return dischargeAmount
    }

    public fun idle() {
        state = "idle"
    }
}
