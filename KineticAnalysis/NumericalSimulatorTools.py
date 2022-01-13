# KineticAnalysis/NumericalSimulatorTools.py


# =======
# Imports
# =======

import math;
import warnings;

from KineticAnalysis.NumericalSimulator import NumericalSimulator;


# =====================
# Convenience Functions
# =====================

def SimulateDynamicRange(decParams, excParams, temperature, excTime, decTime, numSequences = None, dynamicRangeThreshold = 1.0e-3, maxSimulationTimestep = None):
    # TODO: Parameter validation.

    decN, decArrheniusParams, decArrheniusB = decParams;
    excN, excArrheniusParams, excArrheniusB = excParams;

    simulator = NumericalSimulator(
        decN = decN, decArrheniusParams = decArrheniusParams, decArrheniusB = decArrheniusB,
        excN = excN, excArrheniusParams = excArrheniusParams, excArrheniusB = excArrheniusB
        );

    simulator.SetTemperature(temperature);

    simulator.InitialiseTrajectory();

    excAlphaMS, decAlphaMS = [], [];

    sequenceNumber = 0;

    while True:
        if excTime > 0.0:
            simulator.SetExcitation(True);
            simulator.RunTrajectory(runTime = excTime, maxTimestep = maxSimulationTimestep);

            _, simAlphaMS = simulator.GetTrajectory();
            excAlphaMS.append(simAlphaMS[-1]);
        else:
            excAlphaMS.append(0.0);

        if decTime > 0.0:
            simulator.SetExcitation(False);
            simulator.RunTrajectory(runTime = decTime, maxTimestep = maxSimulationTimestep);

            _, simAlphaMS = simulator.GetTrajectory();
            decAlphaMS.append(simAlphaMS[-1]);
        else:
            decAlphaMS.append(excAlphaMS[-1]);

        sequenceNumber = sequenceNumber + 1;

        if numSequences != None:
            if sequenceNumber == numSequences:
                break;
        else:
            if sequenceNumber > 1:
                if math.fabs(excAlphaMS[-1] - excAlphaMS[-2]) < dynamicRangeThreshold and math.fabs(decAlphaMS[-1] - decAlphaMS[-2]) < dynamicRangeThreshold:
                    break;

    if numSequences != None:
        if math.fabs(excAlphaMS[-1] - excAlphaMS[-1]) >= dynamicRangeThreshold or math.fabs(decAlphaMS[-1] - decAlphaMS[-2]) >= dynamicRangeThreshold :
            warnings.warn(
                "numSequences = {0} was insufficient to stabilise the dynamic range below dynamicRangeThreshold = {1:.2e}.".format(numSequences, dynamicRangeThreshold), UserWarning
                );

    simTimes, simAlphaMS = simulator.GetTrajectory();

    return ((simTimes, simAlphaMS), (excAlphaMS[-1], decAlphaMS[-1]), excAlphaMS[-1] - decAlphaMS[-1]);

def SimulateExcDecAlphaMS(decParams, excParams, temperature, excTime, decTime, maxSimulationTimestep = None):
    # TODO: Parameter validation.

    decN, decArrheniusParams, decArrheniusB = decParams;
    excN, excArrheniusParams, excArrheniusB = excParams;

    simulator = NumericalSimulator(
        decN = decN, decArrheniusParams = decArrheniusParams, decArrheniusB = decArrheniusB,
        excN = excN, excArrheniusParams = excArrheniusParams, excArrheniusB = excArrheniusB
        );

    simulator.SetTemperature(temperature);

    simulator.InitialiseTrajectory();

    excAlphaMS = 0.0;

    if excTime > 0.0:
        simulator.SetExcitation(True);
        simulator.RunTrajectory(excTime, maxTimestep = maxSimulationTimestep);

        _, simAlphaMS = simulator.GetTrajectory();
        excAlphaMS = simAlphaMS[-1];

    decAlphaMS = excAlphaMS;

    if decTime > 0.0:
        simulator.SetExcitation(False);
        simulator.RunTrajectory(decTime, maxTimestep = maxSimulationTimestep);

        _, simAlphaMS = simulator.GetTrajectory();
        decAlphaMS = simAlphaMS[-1];

    return (excAlphaMS, decAlphaMS);

def OptimisePulseForDynamicRange(decParams, excParams, temperature, pulseTime, dynamicRangeThreshold = 1.0e-3, convThreshold = 1.0e-4, adjustExcFracThreshold = None, maxSimulationTimestep = None):
    # TODO: Parameter validation.

    excFrac, dynamicRange = 0.0, 0.0;

    excFracStep = 0.1;

    while True:
        # Increase excFrac by excFracStep until the dynamic range stops improving.

        while excFrac + excFracStep < 1.0:
            excFracNew = excFrac + excFracStep;

            _, _, dynamicRangeNew = SimulateDynamicRange(
                decParams, excParams, temperature,
                excFracNew * pulseTime, (1.0 - excFracNew) * pulseTime,
                dynamicRangeThreshold = dynamicRangeThreshold, maxSimulationTimestep = maxSimulationTimestep
                );

            if dynamicRangeNew <= dynamicRange:
                break;

            excFrac, dynamicRange = excFracNew, dynamicRangeNew;

        # Decrease excFrac by excFracStep until the dynamic range stops improving.

        while excFrac > excFracStep:
            excFracNew = excFrac - excFracStep;

            _, _, dynamicRangeNew = SimulateDynamicRange(
                decParams, excParams, temperature,
                excFracNew * pulseTime, (1.0 - excFracNew) * pulseTime,
                dynamicRangeThreshold = dynamicRangeThreshold, maxSimulationTimestep = maxSimulationTimestep
                );

            if dynamicRangeNew < dynamicRange:
                break;

            excFrac, dynamicRange = excFracNew, dynamicRangeNew;

        # Stop when the adjustment step is below the threshold.

        if excFracStep < convThreshold:
            break;

        # Adjust the step.

        excFracStep = excFracStep / 10.0;

    # If adjustExcFracThreshold is set, reduce excFrac until the change in dynamicRange equals or exceeds this value.
    # This optional flag is to allow excFrac to be adjusted to the point where any further increases would be below a measurable threshold.

    if adjustExcFracThreshold != None:
        excFracStep = 0.1;

        while True:
            while excFrac > excFracStep:
                excFracNew = excFrac - excFracStep;

                _, _, dynamicRangeNew = SimulateDynamicRange(
                    decParams, excParams, temperature,
                    excFracNew * pulseTime, (1.0 - excFracNew) * pulseTime,
                    dynamicRangeThreshold = dynamicRangeThreshold, maxSimulationTimestep = maxSimulationTimestep
                    );

                if dynamicRange - dynamicRangeNew >= adjustExcFracThreshold:
                    break;

                excFrac = excFracNew;

            if excFracStep < convThreshold:
                break;

            excFracStep = excFracStep / 10.0;

    _, (excAlphaMS, decAlphaMS), _ = SimulateDynamicRange(
        decParams, excParams, temperature,
        excFrac * pulseTime, (1.0 - excFrac) * pulseTime,
        dynamicRangeThreshold = dynamicRangeThreshold, maxSimulationTimestep = maxSimulationTimestep
        );

    return (excFrac, (excAlphaMS, decAlphaMS), dynamicRange);

def OptimisePulseForCompleteDecay(decParams, excParams, temperature, pulseTime, decayThreshold = 1.0e-3, convThreshold = 1.0e-4, adjustExcFracThreshold = None, maxSimulationTimestep = None):
    # TODO: Parameter validation.

    excFrac, alphaSS = 0.0, 0.0;

    excFracStep = 0.1;

    while True:
        # Increase excFrac until we no longer get complete decay to within decayThreshold.

        while excFrac + excFracStep < 1.0:
            excFracNew = excFrac + excFracStep;

            excAlphaMS, decAlphaMS = SimulateExcDecAlphaMS(
                decParams, excParams, temperature,
                excFracNew * pulseTime, (1.0 - excFracNew) * pulseTime,
                maxSimulationTimestep = maxSimulationTimestep
                );

            if decAlphaMS > decayThreshold or excAlphaMS <= alphaSS:
                break;

            excFrac, alphaSS = excFracNew, excAlphaMS;

        # Decrease excFrac until alphaSS decreases.

        while excFrac > excFracStep:
            excFracNew = excFrac - excFracStep;

            excAlphaMS, decAlphaMS = SimulateExcDecAlphaMS(
                decParams, excParams, temperature,
                excFracNew * pulseTime, (1.0 - excFracNew) * pulseTime,
                maxSimulationTimestep = maxSimulationTimestep
                );

            if decAlphaMS > decayThreshold or excAlphaMS < alphaSS:
                break;

            excFrac, alphaSS = excFracNew, excAlphaMS;

        # Stop when the step is below the threshold.

        if excFracStep < convThreshold:
            break;

        excFracStep = excFracStep / 10.0;

    # By default, the above loop will maximise excFrac within the constraint of the MS population falling below decayThreshold within the remaining pulse time.
    # If adjustExcFracThreshold is set, as in OptimisePulseForDynamicRange() excFrac will be reduced until the change in dynamicRange equals or exceeds the threshold.
    # This is implemented mostly for compatibility with OptimisePulseForDynamicRange().

    if adjustExcFracThreshold != None:
        excFracStep = 0.1;

        while True:
            while excFrac > excFracStep:
                excFracNew = excFrac - excFracStep;

                excAlphaMS, decAlphaMS = SimulateExcDecAlphaMS(
                    decParams, excParams, temperature,
                    excFracNew * pulseTime, (1.0 - excFracNew) * pulseTime,
                    maxSimulationTimestep = maxSimulationTimestep
                    );

                if alphaSS - excAlphaMS >= adjustExcFracThreshold:
                    break;

                excFrac = excFracNew;

            if excFracStep < convThreshold:
                break;

            excFracStep = excFracStep / 10.0;

    excAlphaMS, decAlphaMS = SimulateExcDecAlphaMS(
        decParams, excParams, temperature,
        excFrac * pulseTime, (1.0 - excFrac) * pulseTime,
        maxSimulationTimestep = maxSimulationTimestep
        );

    return (excFrac, (excAlphaMS, decAlphaMS), alphaSS);
