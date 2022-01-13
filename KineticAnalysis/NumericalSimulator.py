# KineticAnalysis/NumericalSimulator.py


# =======
# Imports
# =======

import math;
import warnings;

import numpy as np;

from KineticAnalysis.Constants import UniversalGasConstant;


# ========================
# NumericalSimulator class
# ========================

class NumericalSimulator:
    # ===========
    # Constructor
    # ===========

    def __init__(self, decN = None, decK = None, decArrheniusParams = None, decArrheniusB = None, excN = None, excK = None, excArrheniusParams = None, excArrheniusB = None):
        # Parameter validation.

        if (decN != None and (decN < 0.0 or decN > 4.0)) or (excN != None and (excN < 0.0 or excN > 4.0)):
            raise Exception("Error: If supplied, decN and decK must be between 0 and 4.");

        if decN == None and decK == None:
            raise Exception("Error: Kinetic parameters for at least one of the decay or excitation processes must be supplied.");

        if decN != None:
            if decK == None and decArrheniusParams == None:
                raise Exception("Error: If decN is supplied, one of decK or decArrheniusParams must also be supplied.");
        else:
            if decK != None or decArrheniusParams != None:
                raise Exception("Error: If one of decK or decArrheniusParams is supplied, decN must also be supplied.");

        if excN != None:
            if excK == None and excArrheniusParams == None:
                raise Exception("Error: If excN is supplied, one of excK or excArrheniusParams must also be supplied.");
        else:
            if excK != None or excArrheniusParams != None:
                raise Exception("Error: If one of excK or excArrheniusParams is supplied, excN must also be supplied.");

        if decK != None and decArrheniusParams != None:
            warnings.warn("If both decK and decArrheniusParams are supplied, the Arrhenius parameters take precedence and rate constants must be set by calling the SetTempreature() method.", UserWarning);

        if excK != None and excArrheniusParams != None:
            warnings.warn("If both excK and excArrheniusParams are supplied, the Arrhenius parameters take precedence and rate constants must be set by calling the SetTemperature() method.", UserWarning);

        # If b term(s) for the modified Arrhenius expression(s) for the decay and/or excitation have not been supplied, set them to zero to revert to the standard Arrhenius expression.

        if decArrheniusB == None:
            decArrheniusB = 0.0;

        if excArrheniusB == None:
            excArrheniusB = 0.0;

        # Kinetic parameters for the decay process.

        if decN != None:
            self._decN = decN;

            if decArrheniusParams != None:
                decEA, decLnA = decArrheniusParams;

                self._decK = None;

                # Convert ln(A) -> A and E_A to J mol^-1 before storing internally.

                self._decArrheniusParams = (math.exp(decLnA), decEA * 1000.0, decArrheniusB);
            else:
                self._decK = decK;
                self._decArrheniusParams = None;
        else:
            self._decN = None;
            self._decK = None;

        # Kinetic parameters for the excitation process.

        if excN != None:
            self._excN = excN;

            if excArrheniusParams != None:
                excEA, excLnA = excArrheniusParams;

                self._excK = None;
                self._excArrheniusParams = (math.exp(excLnA), excEA * 1000.0, excArrheniusB);
            else:
                self._excK = excK;
                self._excArrheniusParams = None;
        else:
            self._excN = None;
            self._excK = None;

        self._excActive = excN != None;

        # Initialise other internal fields.

        self._temperature = None;

        self._trajectory = None;

    # =======
    # Methods
    # =======

    def SetTemperature(self, temperature):
        # Sanity checks.

        if temperature == None or temperature <= 0.0:
            raise Exception("Error: temperature must be > 0.");

        if self._decArrheniusParams == None and self._excArrheniusParams == None:
            raise Exception("Error: SetTemperature() can only be called if Arrhenius parameters for at least one of the decay or excitation processes have been supplied.");

        if self._decArrheniusParams == None:
            warnings.warn("Arrhenius parameters for the decay process have not been set - the decay rate constant will not be updated when SetTemperature() is called.");

        if self._excActive and self._excArrheniusParams == None:
            warnings.warn("Arrhenius parameters for the excitation process have not been not set - the excitation rate constant will not be updated when the excitation is active and SetTemperature() is called.");

        # Setting the temperature triggers the update of several internal parameters -> only do it if necessary.

        if temperature != self._temperature:
            # Recalculate rate constants.

            if self._decN != None and self._decArrheniusParams != None:
                decA, decEA, decB = self._decArrheniusParams;
                self._decK = (1.0 + decB * temperature) * decA * math.exp(-1.0 * (decEA / (UniversalGasConstant * temperature)));

            if self._excN != None and self._excArrheniusParams != None:
                excA, excEA, excB = self._excArrheniusParams;
                self._excK = (1.0 + excB * temperature) * excA * math.exp(-1.0 * (excEA / (UniversalGasConstant * temperature)));

            # For certain Arrhenius parameters, in particular for the "modified" Arrhenius model with the temperature-dependent attempt frequency (prefactor), it is possible to end up with negative rate constants.
            # This causes math domain errors when running the trajectory.
            # We counter the problem by taking the absolute value, but we issue a RuntimeWarning, since this definitely constitutes a "dubious runtime feature" (!).

            if self._decK != None and self._decK < 0.0:
                warnings.warn("The current Arrhenius parameters produce k_dec = {0:.2e} < 0.0 for T = {1:.1f}; the absolute value will be taken to avoid math domain errors.".format(self._decK, temperature), RuntimeWarning);

                self._decK = math.fabs(self._decK);

            elif self._excK != None and self._excK < 0.0:
                warnings.warn("The current Arrhenius parameters produce k_exc = {0:.2e} < 0.0 for T = {1:.1f}; the absolute value will be taken to avoid math domain errors.".format(self._excK, temperature), RuntimeWarning);

                self._excK = math.fabs(self._excK);

            # Store updated temperature.

            self._temperature = temperature;

    def SetExcitation(self, excitationActive):
        if excitationActive and self._excN == None:
            raise Exception("Error: Excitation can only be modelled if kinetic parameters are supplied at initialisation.");

        self._excActive = excitationActive;

    def InitialiseTrajectory(self, initialAlphaMS = 0.0):
        self._trajectory = (
            np.array([0.0], dtype = np.float64),
            np.array([initialAlphaMS], dtype = np.float64)
            );

    def RunTrajectory(self, runTime = None, updateTimestepCycles = None, maxTimestep = None):
        # If the user has not supplied decay parameters, and has switched excitation off, there may be no active processes to simulate.

        if self._decN == None and not self._excActive:
            raise Exception("Error: No active processes to simulate.");

        # Check the trajectory has been initialised.

        if self._trajectory == None:
            raise Exception("Error: InitialiseTrajectory() must be called before Run().");

        decN, decK = self._decN, self._decK;
        excN, excK = self._excN, self._excK;

        # If a JMAK exponent for the decay/excitation process has been supplied, but the corresponding rate constant has not been set, it should mean the user supplied Arrhenius parameters at construction, but has not called SetTemperature() to initialise the rate constants.

        if decN != None and decK == None:
            raise Exception("Error: If Arrhenius parameters for the decay process are supplied at construction time, SetTemperature() must be called before RunTrajectory() to set the rate constant.");

        excActive = self._excActive;

        # excActive should only be set if kinetic parameters for the excitation have been supplied; however, if Arrhenius parameters were supplied, we need to check whether the temperature has been set.

        if excActive and (excN != None and excK == None):
            raise Exception("Error: If the excitation process is active, and Arrhenius parameters for the process are supplied at construction time, SetTemperature() must be called before RunTrajectory() to set the rate constant.");

        # If the excitation rate is very small, it can cause math domain errors; to avoid this happening, if the excitation rate is below a certain threshold, we disable the excitation process.

        if excActive and excK < NumericalSimulator.ExcActiveThreshold:
            warnings.warn("The current k_exc = {0:.2e} is less than the ExcActiveThreshold = {1:.2e}; the excitation process will be disabled to avoid math domain errors.".format(self._excK, NumericalSimulator.ExcActiveThreshold), RuntimeWarning);

            excActive = False;

        # Sanity checks.

        if runTime != None and runTime <= 0.0:
            raise Exception("Error: If supplied, runTime must be > 0.");

        if updateTimestepCycles != None and updateTimestepCycles <= 0:
            raise Exception("Error: If supplied, updateTimestepCycles must be an integer > 0.");

        # If updateTimestepCycles is not set, set it to 1 (i.e. the timestep will be updated every cycle).

        if updateTimestepCycles == None:
            updateTimestepCycles = 1;

        # If maxTimestep is not set, set it to the default value.

        if maxTimestep == None:
            maxTimestep = NumericalSimulator.MaxTimestep;

        # Load the initial MS occupation.

        _, trajectoryAlphaMS = self._trajectory;
        alphaMS = trajectoryAlphaMS[-1];

        # Variable to hold the timestep.

        timestep = None;

        # Variables to keep track of the time and the number of cycles.

        time = 0.0;
        numCycles = 0;

        # Lists to collect the trajectory.

        tData, alphaMSData = [], [];

        # Run the simulation.
        
        while True:
            # Calculate the effective decay/excitation "time" (i.e. where on the respective JMAK curves the current occupation is).

            decT1, excT1 = None, None;

            if decN != None and alphaMS > 0.0:
                decT1 = math.pow((-1.0 / decK) * math.log(alphaMS), 1.0 / decN);

            if excActive and alphaMS < 1.0:
                excT1 = math.pow((-1.0 / excK) * math.log(1.0 - alphaMS), 1.0 / excN);

            # Recalculate the timestep if required.

            if timestep == None or numCycles % updateTimestepCycles == 0:
                # Default to KineticsSimulation.MaxTimestep.

                timestep = maxTimestep;

                # If the decay process is active, calculate the timestep required for alphaMS -> alphaMS - KineticsSimulation.MaxDeltaAlphaMSPerTimestep.
                # If this is shorter than the current timestep, update it.

                if decT1 != None and alphaMS - NumericalSimulator.MaxDeltaAlphaMSPerTimestep > 0.0:
                    decT2 = math.pow((-1.0 / decK) * math.log(alphaMS - NumericalSimulator.MaxDeltaAlphaMSPerTimestep), 1.0 / decN);

                    timestep = min(timestep, decT2 - decT1);

                # If the excitation process is active, calculate the timestep required for alphaMS -> alphaMS + KineticsSimulation.MaxDeltaAlphaMSPerTimestep.
                # Again, if this is shorter than the current timestep, update it.

                if excT1 != None and alphaMS + NumericalSimulator.MaxDeltaAlphaMSPerTimestep < 1.0:
                    excT2 = math.pow((-1.0 / excK) * math.log(1.0 - (alphaMS + NumericalSimulator.MaxDeltaAlphaMSPerTimestep)), 1.0 / excN);

                    timestep = min(timestep, excT2 - excT1);

            # Clamp the timestep so we don't go over the simulation time (if set).

            if runTime != None:
                timestep = min(timestep, runTime - time);

            # Calculate the changes in alphaMS due to excitation and/or decay and update.

            alphaMSNew = alphaMS;

            if excActive and excT1 != None:
                alphaMSNew = alphaMSNew + (1.0 - math.exp(-1.0 * excK * (excT1 + timestep) ** excN)) - (1.0 - math.exp(-1.0 * excK * excT1 ** excN));

            if decT1 != None:
                alphaMSNew = alphaMSNew + math.exp(-1.0 * decK * (decT1 + timestep) ** decN) - math.exp(-1.0 * decK * decT1 ** decN);

            # Clamp alphaMS to the range [0, 1].

            alphaMSNew = max(0.0, alphaMSNew);
            alphaMSNew = min(alphaMSNew, 1.0);

            time = time + timestep;
            alphaMS = alphaMSNew;

            tData.append(time);
            alphaMSData.append(alphaMS);

            # Check the break criteria.

            breakCurrent = False;
            
            if runTime != None:
                if time >= runTime:
                    breakCurrent = True;
            else:
                if math.fabs(alphaMSNew - alphaMS) < NumericalSimulator.EquilibrationConvergenceCriterion:
                    breakCurrent = True;

            if breakCurrent:
                break;

        trajectoryTime, _ = self._trajectory;

        trajectoryTimeUpdate = np.array(tData, dtype = np.float64) + trajectoryTime[-1];
        trajecotryAlphaMSUpdate = np.array(alphaMSData, dtype = np.float64);

        self._trajectory = (
            np.concatenate(
                (trajectoryTime, trajectoryTimeUpdate)
                ),
            np.concatenate(
                (trajectoryAlphaMS, trajecotryAlphaMSUpdate)
                )
            );

    def GetTrajectory(self, dumpCycles = 1):
        if self._trajectory == None:
            raise Exception("Error: A simulation must have been initialised and run before calling GetTrajectory().");

        trajectoryTime, trajectoryAlphaMS = self._trajectory;

        return (trajectoryTime[::dumpCycles], trajectoryAlphaMS[::dumpCycles])

    # =========
    # Constants
    # =========

    MaxTimestep = 1.0;
    MaxDeltaAlphaMSPerTimestep = 1.0e-4;

    EquilibrationConvergenceCriterion = 1.0e-8;

    ExcActiveThreshold = 1.0e-16;
