package org.opendc.simulator.compute.energy;

import java.util.List;
import java.util.ArrayList;

public class EnergyUtils {
    public static List<Double> generateGreenEnergyProfile(int steps, double maxGreenEnergy) {
        List<Double> profile = new ArrayList<>();
        for (int i = 0; i < steps; i++) {
            profile.add(Math.random() * maxGreenEnergy);
        }
        return profile;
    }
}
