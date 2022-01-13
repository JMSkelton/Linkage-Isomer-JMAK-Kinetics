# KineticAnalysis/KineticDataFitter.py


# =======
# Imports
# =======

import csv;
import itertools;
import math;
import warnings;

import numpy as np;

from scipy.stats import mode;
from scipy.optimize import minimize;

from KineticAnalysis.Constants import UniversalGasConstant;
from KineticAnalysis.NumericalSimulator import NumericalSimulator;
from KineticAnalysis.Utilities import OpenForCSVWriter;


# =======================
# KineticDataFitter class
# =======================

class KineticDataFitter:
    # ===========
    # Constructor
    # ===========

    def __init__(self, decCurves = None, excCurves = None, ssOccupations = None, fitType = 'SimultaneousN'):
        # Check data and parameters.

        if decCurves == None and excCurves == None:
            raise Exception("Error: At least one of decCurves and excCurves must be supplied.");

        if ssOccupations != None and (decCurves == None or excCurves == None):
            raise Exception("Error: Steady-state data can only be analysed in conjunction with decay and excitation curves.");

        curveDataPointsOK = True;

        if decCurves != None:
            temperatures, curves = decCurves;

            if len(temperatures) < 3:
                warnings.warn("To perform Arrhenius fits, decay curves must be measured at at least three temperatures.", UserWarning);

            for times, _ in curves:
                if (fitType == 'SimultaneousN' and len(times) < 4) or (fitType == 'Independent' and len(times) < 5):
                    curveDataPointsOK = False;

        if excCurves != None:
            temperatures, curves = excCurves;

            if len(temperatures) < 3:
                warnings.warn("To perform Arrhenius fits, excitation curves must be measured at at least three temperatures.", UserWarning)

            for times, _ in curves:
                if (fitType == 'SimultaneousN' and len(times) < 4) or (fitType == 'Independent' and len(times) < 5):
                    curveDataPointsOK = False;

        if not curveDataPointsOK:
            raise Exception("Error: Decay/excitation curves must have at least four (fitType = 'SimultaneousN') or five (fitType = 'Independent') data points.");

        if fitType not in KineticDataFitter.FitTypesDescriptions:
            raise Exception("Error: Unsupported fitType '{0}'.".format(fitType));

        if fitType == 'Independent':
            warnings.warn("Arrhenius fits can only be performed for fitType = 'SimultaneousN'.");

        # Store variables.

        self._decCurves = decCurves;
        self._excCurves = excCurves;

        self._ssOccupations = ssOccupations;

        self._fitType = fitType;

        # Initialise results fields.

        self._decCurveFits = None;
        self._excCurveFits = None;

        self._decArrheniusParams = None;
        self._excArrheniusParams = None;

        self._ssArrheniusParams = None;

    # ===============
    # Private Methods
    # ===============

    def _CurvesFitFunction(self, dataType, params):
        # Load experimental data.

        curves = None;

        if dataType == 'dec':
            _, curves = self._decCurves;
        elif dataType == 'exc':
            _, curves = self._excCurves;

        # Chain the fitted MS occupation curves into a single list.

        alphaMSFit = [];

        if self._fitType == 'Independent':
            for i, (times, _) in enumerate(curves):
                alpha0, alphaF, k, n = params[i * 4:(i + 1) * 4];

                alphaMSFit.extend(
                    JMAKEquation(times, alpha0, alphaF, k, n)
                    );

        elif self._fitType == 'SimultaneousN':
            n = params[0];

            for i, (times, _) in enumerate(curves):
                alpha0, alphaF, k = params[i * 3 + 1:(i + 1) * 3 + 1];

                alphaMSFit.extend(
                    JMAKEquation(times, alpha0, alphaF, k, n)
                    );

        elif self._fitType == 'FixedN':
            for i, (times, _) in enumerate(curves):
                alpha0, alphaF, k = params[i * 3:(i + 1) * 3];

                alphaMSFit.extend(
                    JMAKEquation(times, alpha0, alphaF, k, 1.0)
                    );

        # Chain the experimental MS occupation curves into a single list.

        alphaMSExp = [];

        for _, alphaMS in curves:
            alphaMSExp.extend(alphaMS);

        # Return the root mean-square (RMS) error.

        return math.sqrt(np.mean(np.subtract(alphaMSExp, alphaMSFit) ** 2));

    def _CurvesFitFunctionDec(self, params):
        # Dispatch to _CurvesFitFunction with decay data.

        return self._CurvesFitFunction('dec', params);

    def _CurvesFitFunctionExc(self, params):
        # Dispatch to _CurvesFitFunction with excitation data.

        return self._CurvesFitFunction('exc', params);

    def _ArrheniusSSFitFunction(self, params):
        (decT, decCurves), (excT, excCurves) = self._decCurves, self._excCurves;

        decCurveFits, excCurveFits = self._decCurveFits, self._excCurveFits;

        _, _, _, decN, _ = decCurveFits[0];
        _, _, _, excN, _ = excCurveFits[0];

        ssT, ssAlphaMS = self._ssOccupations;

        decEA, decLnA, decB, excEA, excLnA, excB = params;

        # Prepare data to optimise against.

        fitData = [];

        # Setup a NumericalSimulator with the current kinetic parameters.

        simulator = NumericalSimulator(
            decN = decN, decArrheniusParams = (decEA, decLnA), decArrheniusB = decB,
            excN = excN, excArrheniusParams = (excEA, excLnA), excArrheniusB = excB
            );

        # Average deviation in measured and simulated MS occupations for data points on each of the experimental decay/excitation curves.

        for excOn, temperatures, curves in (False, decT, decCurves), (True, excT, excCurves):
            for t, (times, alphaMS) in zip(temperatures, curves):
                simulator.SetTemperature(t);
                simulator.SetExcitation(excOn);

                simulator.InitialiseTrajectory(initialAlphaMS = alphaMS[0]);

                for i in range(1, len(times)):
                    simulator.RunTrajectory(times[i] - times[i - 1]);

                    _, simAlphaMS = simulator.GetTrajectory();

                    fitData.append(
                        math.fabs(alphaMS[i] - simAlphaMS[-1])
                        );

        # Average deviation in measured and simulated steady-state occupations.

        for t, alphaMS in zip(ssT, ssAlphaMS):
            simulator.SetTemperature(t);

            simulator.InitialiseTrajectory();
            simulator.RunTrajectory();

            _, simAlphaMS = simulator.GetTrajectory();

            fitData.append(
                math.fabs(alphaMS - simAlphaMS[-1])
                );

        # Return the root mean-square (RMS) error between the experimental and fitted (simulated) data points.

        rms = math.sqrt(np.mean(np.power(fitData, 2)));

        return rms;

    # ==============
    # Pubilc Methods
    # ==============

    def FitCurves(self, tolerance = None, maxCycles = None, tryAlternativeAlgorithms = True):
        # If tolerance or maxCycles are not set, use the default values.

        if tolerance == None:
            tolerance = KineticDataFitter.DefaultToleranceFitCurves;

        if maxCycles == None:
            maxCycles = KineticDataFitter.DefaultMaxCycles;

        # Sanity checks.

        if tolerance <= 0.0:
            raise Exception("Error: If supplied, tolerance must be greater than zero.");

        if maxCycles <= 0:
            raise Exception("Error: If supplied, maxCycles must be greater than zero.");

        # Group fits into a "task list" to save typing.

        taskList = [];

        if self._decCurves != None:
            taskList.append('dec');

        if self._excCurves != None:
            taskList.append('exc');

        # Loop over fits.

        for dataType in taskList:
            # Load kinetic curves to analyse.

            curves = None;

            if dataType == 'dec':
                _, curves = self._decCurves;
            elif dataType == 'exc':
                _, curves = self._excCurves;

            # The JMAK fits are very sensitive to the initial parameters.
            # It is straightforward to derive sensible guesses for alpha_0 and alpha_F from the data, but not for k and n.
            # To get round this, we sweep a range of initial parameters and choose those which give the lowest root-mean-square (RMS) error.
            # This should (hopefully!) allow the best initial guess of n and k for subsequent fitting to be found.

            initialFits = [];

            for times, alphaMS in curves:
                # Temporary function for obtaining the RMS error of a JMAK fit with the supplied parameters to the experimental data.

                def _JMAKRMSTemp(alpha0, alphaF, k, n):
                    return math.sqrt(np.mean((alphaMS - JMAKEquation(times, alpha0, alphaF, k, n)) ** 2));

                alpha0Init, alphaFInit = alphaMS[0], alphaMS[-1];

                kInitOpt, nInitOpt = KineticDataFitter.FitKTrials[0], KineticDataFitter.FitNTrials[0];
                rmsErrorInitOpt = _JMAKRMSTemp(alpha0Init, alphaFInit, kInitOpt, nInitOpt);

                # Loop over a predetermined set of values of n.

                for nInit in KineticDataFitter.FitNTrials:
                    # Loop over a predetermined set of values of k.

                    for kInit in KineticDataFitter.FitKTrials:
                        fitResultCurrent = _JMAKRMSTemp(alpha0Init, alphaFInit, kInit, nInit);

                        if fitResultCurrent < rmsErrorInitOpt:
                            kInitOpt, nInitOpt = kInit, nInit;
                            rmsErrorInitOpt = fitResultCurrent;

                initialFits.append((alpha0Init, alphaFInit, kInitOpt, nInitOpt));

            # Build a set of initial parameter guesses and a set of bounds on the parameters depending on the fit type.

            pInit, bounds = None, None;

            if self._fitType == 'Independent':
                pInit = [];

                for item in initialFits:
                    pInit.extend(item);

                bounds = [(0.0, 1.0), (0.0, 1.0), (None, None), (None, None)] * len(curves);

            elif self._fitType == 'SimultaneousN':
                pInit = [mode([n for _, _, _, n in initialFits]).mode];

                for alpha0Init, alphaFInit, kInit, _ in initialFits:
                    pInit = pInit + [alpha0Init, alphaFInit, kInit];

                bounds = [(None, None)] + [(0.0, 1.0), (0.0, 1.0), (None, None)]* len(curves);

            elif self._fitType == 'FixedN':
                pInit = [];

                for alpha0Init, alphaFInit, kInit, _ in initialFits:
                    pInit = pInit + [alpha0Init, alphaFInit, kInit];

                bounds = [(0.0, 1.0), (0.0, 1.0), (None, None)] * len(curves);

            else:
                raise NotImplementedError("Error: fitType '{0}' is not implemented.".format(self._fitType));

            # Perform the fit.

            fitFunction = None;

            if dataType == 'dec':
                fitFunction = self._CurvesFitFunctionDec;
            elif dataType == 'exc':
                fitFunction = self._CurvesFitFunctionExc;

            result = minimize(
                fitFunction, pInit, bounds = bounds,
                method = KineticDataFitter.MinimisationFunctions[0], tol = tolerance, options = { 'maxiter' : maxCycles }
                );

            # This is implemented mainly for testing purposes.
            # If tryAlternativeAlgorithms is set, iterate over other minimisation algorithms that support constraints, perform the minimisation with each one, and print an information message if it yields a smaller RMS error.

            if tryAlternativeAlgorithms:
                for function in KineticDataFitter.MinimisationFunctions[1:]:
                    resultNew = minimize(
                        fitFunction, pInit, bounds = bounds, method = function,
                        tol = tolerance, options = { 'maxiter' : maxCycles }
                        );

                    if resultNew.fun < result.fun:
                        print("INFO: KineticDataFitter.FitCurves(): Minimising with '{0}' gives RMS {1:.5f} < current {2} w/ RMS {3:.5f}".format(function, resultNew.fun, KineticDataFitter.MinimisationFunctions[0], result.fun));

            # Gather the fit results into one set of fitted (alpha0, alphaF, k, n) parameters for each decay curve.

            fitResults = [];

            if self._fitType == 'Independent':
                for i, (times, alphaMS) in enumerate(curves):
                    alpha0Opt, alphaFOpt, kOpt, nOpt = result.x[i * 4:(i + 1) * 4];
                    rmsError = math.sqrt(np.average(np.subtract(alphaMS, JMAKEquation(times, alpha0Opt, alphaFOpt, kOpt, nOpt)) ** 2));

                    fitResults.append(
                        (alpha0Opt, alphaFOpt, kOpt, nOpt, rmsError)
                        );

            elif self._fitType == 'SimultaneousN':
                nOpt = result.x[0];

                for i, (times, alphaMS) in enumerate(curves):
                    alpha0Opt, alphaFOpt, kOpt = result.x[i * 3 + 1:(i + 1) * 3 + 1];
                    rmsError = math.sqrt(np.average(np.subtract(alphaMS, JMAKEquation(times, alpha0Opt, alphaFOpt, kOpt, nOpt)) ** 2));

                    fitResults.append(
                        (alpha0Opt, alphaFOpt, kOpt, nOpt, rmsError)
                        );

            elif self._fitType == 'FixedN':
                for i, (times, alphaMS) in enumerate(curves):
                    alpha0Opt, alphaFOpt, kOpt = result.x[i * 3:(i + 1) * 3];
                    rmsError = math.sqrt(np.average(np.subtract(alphaMS, JMAKEquation(times, alpha0Opt, alphaFOpt, kOpt, 1.0)) ** 2));

                    fitResults.append(
                        (alpha0Opt, alphaFOpt, kOpt, 1.0, rmsError)
                        );

            # Store the fit results.

            if dataType == 'dec':
                self._decCurveFits = fitResults;
            elif dataType == 'exc':
                self._excCurveFits = fitResults;

        # Return the fit results.

        fitResults = { };

        if self._decCurveFits != None:
            fitResults['dec'] = self._decCurveFits;

        if self._excCurveFits != None:
            fitResults['exc'] = self._excCurveFits;

        return fitResults;

    def PerformArrheniusAnalysis(self):
        # Check the fit type - an Arrhenius analysis can only be performed with the 'SimultaneousN' option.

        if self._fitType != 'SimultaneousN' and self._fitType != 'FixedN':
            raise Exception("Error: An Arrhenius analysis can only be performed for the 'SimultaneousN' and 'FixedN' fit types.");

        # Check FitCurves() has been called to fit the experimental data to obtain rate constants.

        if self._decCurveFits == None and self._decCurveFits == None:
            raise Exception("Error: FitCurves() must be called before PerformArrheniusAnalysis().");

        # Group fits into a "task list" to save typing.

        taskList = [];

        if self._decCurveFits != None:
            taskList.append('dec');

        if self._excCurveFits != None:
            taskList.append('exc');

        # Loop over fits.

        for dataType in taskList:
            # Load data to analyse.

            temperatures, fitResults = None, None;

            if dataType == 'dec':
                temperatures, _ = self._decCurves;
                fitResults = self._decCurveFits;
            elif dataType == 'exc':
                temperatures, _ = self._excCurves;
                fitResults = self._excCurveFits;

            if len(temperatures) < 3:
                raise Exception("Error: Arrhenius fits require decay and/or excitation curves to have been collected at at least three temperatures.");

            # Build a fitting dataset inverse temperatures and ln(k) values.

            invT, lnK = [], [];

            for temperature, (_, _, kOpt, _, _) in zip(temperatures, fitResults):
                invT.append(1.0 / temperature);
                lnK.append(math.log(kOpt));

            invT, lnK = np.array(invT, dtype = np.float64), np.array(lnK, dtype = np.float64);

            # Fit to a straight line to obtain the gradient and intercept, and calculate the RMS error of the fit.

            m, c = np.polyfit(invT, lnK, deg = 1);
            rmsError = math.sqrt(np.mean((lnK - (m * invT + c)) ** 2));

            # Calculate the activation energy (E_A) and the pre-exponential factor (A) from the fit parameters.

            eA = (-1.0 * UniversalGasConstant * m) / 1000.0;

            # Store the fit result.

            if dataType == 'dec':
                self._decArrheniusParams = (eA, c, rmsError);
            elif dataType == 'exc':
                self._excArrheniusParams = (eA, c, rmsError);

        # Return the fit results.

        fitResults = { };

        if self._decArrheniusParams != None:
            fitResults['dec'] = self._decArrheniusParams;

        if self._excArrheniusParams != None:
            fitResults['exc'] = self._excArrheniusParams;

        return fitResults;

    def RefitArrheniusWithSSData(self, tolerance = None, maxCycles = None, sweepDec = True, sweepExc = False, tryAlternativeAlgorithms = True):
        # Check Arrhenius analyses have been performed for both excitation and decay.
        # TODO: Other checks encompassed by this.

        if self._decArrheniusParams == None:
            if self._excArrheniusParams == None:
                raise Exception("Error: PerformArrheniusAnalysis() must be called before RefitArrheniusWithSSData().");
            else:
                raise Exception("Error: Analysing steady-state data requires Arrhenius parameters for both excitation and decay processes.");

        # Sanity check.

        if self._ssOccupations == None:
            raise Exception("Error: Steady-state occupations must be passed at initialisation.");

        # If tolerance or maxCycles are not set, use the default values.

        if tolerance == None:
            tolerance = KineticDataFitter.DefaultToleranceArrheniusRefit;

        if maxCycles == None:
            maxCycles = KineticDataFitter.DefaultMaxCycles;

        # More sanity checks.

        if tolerance <= 0.0:
            raise Exception("Error: If supplied, tolerance must be greater than zero.");

        if maxCycles <= 0:
            raise Exception("Error: If supplied, maxCycles must be greater than zero.");

        # Use the Arrhenius parameters fitted from the rate constant as an initial guess.

        decEA, decLnA, _ = self._decArrheniusParams;
        excEA, excLnA, _ = self._excArrheniusParams;

        # The minimisation is quite sensitive to the initial parameter guesses, since it will typically start close to a local minimum.
        # To get round this we try a range of initial parameters and choose those which give the lowest root-mean-square (RMS) error.
        # Unlike with the JMAK fits performed in FitCurves(), however, we perform a complete minimisation for each combination.

        trialCombinations = [];

        if sweepDec:
            trialCombinations.append(
                [decLnA * scale for scale in KineticDataFitter.RefitArrheniusLnAScaleTrials]
                );

            trialCombinations.append(
                KineticDataFitter.RefitArrheniusBTrials
                );
        else:
            trialCombinations = trialCombinations + [[decLnA], [0.0]];

        if sweepExc:
            trialCombinations.append(
                [excLnA * scale for scale in KineticDataFitter.RefitArrheniusLnAScaleTrials]
                );

            trialCombinations.append(
                KineticDataFitter.RefitArrheniusBTrials
                );
        else:
            trialCombinations = trialCombinations + [[excLnA], [0.0]];

        # Constrain the b terms to be > 0.

        bounds = [(None, None), (None, None), (0.0, None)] * 2;

        pInitOpt, resultOpt = None, None;

        for decLnAInit, decBInit, excLnAInit, excBInit in itertools.product(*trialCombinations):
            print("Minimisation w/ decLnA = {0:.2e}, decB = {1:.2e}, excLnA = {2:.2e}, excB = {3:.2e}".format(decLnAInit, decBInit, excLnAInit, excBInit));

            pInit = [
                decEA, decLnAInit, decBInit,
                excEA, excLnAInit, excBInit
                ];

            result = minimize(
                self._ArrheniusSSFitFunction, pInit, bounds = bounds,
                method = KineticDataFitter.MinimisationFunctions[0], tol = tolerance, options = { 'maxiter' : maxCycles }
                );

            print("  -> RMS = {0:.6f}".format(result.fun));

            if pInitOpt == None or result.fun < resultOpt.fun:
                pInitOpt = pInit;
                resultOpt = result;

        # As in FitCurves(), if tryAlternativeAlgorithms is set, iterate over other minimisation algorithms and print an information message if it yields a smaller RMS error.

        if tryAlternativeAlgorithms:
            for function in KineticDataFitter.MinimisationFunctions[1:]:
                resultNew = minimize(
                    self._ArrheniusSSFitFunction, pInitOpt, bounds = bounds,
                    method = function, tol = tolerance, options = { 'maxiter' : maxCycles }
                    );

                if resultNew.fun < resultOpt.fun:
                    print("INFO: KineticDataFitter.RefitWithSSData(): Minimising with '{0}' gives error sum {1:.5f} < current {2} w/ error sum {3:.5f}".format(function, resultNew.fun, KineticDataFitter.MinimisationFunctions[0], result.fun));

        # Store and return the new fit results.

        decEANew, decLnANew, decB, excEANew, excLnANew, excB = resultOpt.x;

        print("Best-fit parameters:");
        print(decEANew, decLnANew, decB);
        print(excEANew, excLnANew, excB);
        print(resultOpt.fun);

        kineticParamsNew = (
            (decEANew, decLnANew, decB),
            (excEANew, excLnANew, excB)
            );

        self._ssArrheniusParams = kineticParamsNew;

        return kineticParamsNew;

    def SaveFittingResults(self, filePath):
        # Check that at least FitCurves() has been called.

        if self._decCurveFits == None and self._excCurveFits == None:
            raise Exception("Error: FitCurves() must be called before SaveFittingResults().");

        with OpenForCSVWriter(filePath) as outputWriter:
            outputWriterCSV = csv.writer(outputWriter, delimiter = ',', quotechar = '\"', quoting = csv.QUOTE_ALL);

            # Write the fit type and its "accessible description".

            outputWriterCSV.writerow(["Fit Type: {0} ('{1}')".format(KineticDataFitter.FitTypesDescriptions[self._fitType], self._fitType)]);
            outputWriterCSV.writerow([]);

            # Group results into a "task list" to save typing.

            taskList = [];

            if self._decCurveFits != None:
                taskList.append('dec');

            if self._excCurveFits != None:
                taskList.append('exc');

            # Loop over tasks.

            for dataType in taskList:
                # Load data to write out.

                temperatures, curves = None, None;
                fitResults, arrheniusParams = None, None;
                blockHeader = None;

                if dataType == 'dec':
                    temperatures, curves = self._decCurves;
                    fitResults = self._decCurveFits;

                    if self._decArrheniusParams != None:
                        arrheniusParams = self._decArrheniusParams;

                    blockHeader = "Analysis of Decay Curves";

                elif dataType == 'exc':
                    temperatures, curves = self._excCurves;
                    fitResults = self._excCurveFits;

                    if self._excArrheniusParams != None:
                        arrheniusParams = self._excArrheniusParams;

                    blockHeader = "Analysis of Excitation Curves";

                outputWriterCSV.writerow([blockHeader]);
                outputWriterCSV.writerow([]);

                # Write out the fit parameters.

                outputWriterCSV.writerow(["T [K]", "alpha_0", "alpha_F", "k [s^-n]", "n", "RMS"]);

                for temperature, (alpha0, alphaF, k, n, rmsError) in zip(temperatures, fitResults):
                    outputWriterCSV.writerow([temperature, alpha0, alphaF, k, n, rmsError]);

                outputWriterCSV.writerow([]);

                # Include the Arrhenius fit parameters, if the analysis has been performed.

                if arrheniusParams != None:
                    outputWriterCSV.writerow(["Arrhenius Analysis"]);
                    outputWriterCSV.writerow([]);

                    eA, lnA, rmsError = arrheniusParams;

                    outputWriterCSV.writerow(["E_A [kJ mol^-1]", "ln(A)", "RMS Error"]);
                    outputWriterCSV.writerow([eA, lnA, rmsError]);

                    outputWriterCSV.writerow([]);

                    # Write out the raw and fitted rate constants.

                    outputWriterCSV.writerow(["1/T [K^-1]", "ln(k) (Exp.)", "ln(k) (Fit)"]);

                    for temperature, (_, _, k, _, _) in zip(temperatures, fitResults):
                        outputWriterCSV.writerow([1.0 / temperature, math.log(k), lnA - ((eA * 1000.0) / (UniversalGasConstant * temperature))]);

                    outputWriterCSV.writerow([]);

                # Write out the raw and fitted curves.

                headerRow = [];

                for temperature in temperatures:
                    headerRow = headerRow + ["T = {0:.2f} K".format(temperature), "", ""];

                outputWriterCSV.writerow(headerRow);

                outputWriterCSV.writerow(["t [s]", r"\alpha_MS (Exp.)", r"\alpha_MS (Fit)"] * len(temperatures));

                # Calculate the fitted curves to output alongside the experimental data.

                fittedCurves = [
                    JMAKEquation(times, alpha0, alphaF, k, n)
                        for (times, _), (alpha0, alphaF, k, n, _) in zip(curves, fitResults)
                    ];

                # The decay curves may not all have the same number of values.

                numRows = max(len(times) for times, _ in curves);

                for i in range(0, numRows):
                    dataRow = [];

                    for (times, alphaMS), alphaMSFit in zip(curves, fittedCurves):
                        if len(times) > i:
                            dataRow = dataRow + [times[i], alphaMS[i], alphaMSFit[i]];
                        else:
                            dataRow = dataRow + ["", "", ""];

                    outputWriterCSV.writerow(dataRow);

                outputWriterCSV.writerow([]);

            if self._ssArrheniusParams != None:
                outputWriterCSV.writerow(["Arrhenius Refit w/ Pseudo Steady-State Data"])
                outputWriterCSV.writerow([]);

                (decEA1, decLnA1, _) = self._decArrheniusParams;
                (excEA1, excLnA1, _) = self._excArrheniusParams;

                (decEA2, decLnA2, decB2), (excEA2, excLnA2, excB2) = self._ssArrheniusParams;

                outputWriterCSV.writerow(["", "E_A [kJ mol^-1]", "ln(A)", "b [K^-1]"]);

                outputWriterCSV.writerow(["Decay (Initial)", decEA1, decLnA1, '-']);
                outputWriterCSV.writerow(["Decay (Revised)", decEA2, decLnA2, decB2]);

                outputWriterCSV.writerow(["Excitation (Initial)", excEA1, excLnA1, '-']);
                outputWriterCSV.writerow(["Excitation (Revised)", excEA2, excLnA2, excB2]);

                outputWriterCSV.writerow([]);

                (decT, _), (excT, _) = self._decCurves, self._excCurves;
                decCurveFits, excCurveFits = self._decCurveFits, self._excCurveFits;

                outputWriterCSV.writerow(["1/T [K^-1]", "ln(k_dec) (Exp.)", "ln(k_dec) (Init.)", "ln(k_dec) (Rev.)"]);

                for i, t in enumerate(decT):
                    _, _, k, _, _ = decCurveFits[i];

                    lnK1 = decLnA1 - (decEA1 * 1000.0) / (UniversalGasConstant * t);
                    lnK2 = math.log(1.0 + decB2 * t) + decLnA2 - (decEA2 * 1000.0) / (UniversalGasConstant * t);

                    outputWriterCSV.writerow([1.0 / t, math.log(k), lnK1, lnK2]);

                outputWriterCSV.writerow([]);

                outputWriterCSV.writerow(["1/T [K^-1]", "ln(k_exc) (Exp.)", "ln(k_exc) (Init.)", "ln(k_exc) (Rev.)"]);

                for i, t in enumerate(excT):
                    _, _, k, _, _ = excCurveFits[i];

                    lnK1 = excLnA1 - (excEA1 * 1000.0) / (UniversalGasConstant * t);
                    lnK2 = math.log(1.0 + excB2 * t) + excLnA2 - (excEA2 * 1000.0) / (UniversalGasConstant * t);

                    outputWriterCSV.writerow([1.0 / t, math.log(k), lnK1, lnK2]);

                outputWriterCSV.writerow([]);

                ssT, ssAlphaMS = self._ssOccupations;

                _, _, _, decN, _ = decCurveFits[0];
                _, _, _, excN, _ = excCurveFits[0];

                simulator1 = NumericalSimulator(
                    decN = decN, decArrheniusParams = (decEA1, decLnA1),
                    excN = excN, excArrheniusParams = (excEA1, excLnA1)
                    );

                simulator2 = NumericalSimulator(
                    decN = decN, decArrheniusParams = (decEA2, decLnA2), decArrheniusB = decB2,
                    excN = excN, excArrheniusParams = (excEA2, excLnA2), excArrheniusB = excB2
                    );

                outputWriterCSV.writerow(["T [K]", r"\alpha_SS (Exp)", r"\alpha_SS (Init.)", r"\alpha_SS (Rev.)"]);

                for t, ssAlphaMS in zip(ssT, ssAlphaMS):
                    simulator1.SetTemperature(t);

                    simulator1.InitialiseTrajectory();
                    simulator1.RunTrajectory();

                    _, alphaMS1 = simulator1.GetTrajectory();

                    simulator2.SetTemperature(t);

                    simulator2.InitialiseTrajectory();
                    simulator2.RunTrajectory();

                    _, alphaMS2 = simulator2.GetTrajectory();

                    outputWriterCSV.writerow([t, ssAlphaMS, alphaMS1[-1], alphaMS2[-1]]);

                outputWriterCSV.writerow([]);

    # =========
    # Constants
    # =========

    FitKTrials = [1.0e2, 1.0e1, 1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8];

    FitNTrials = [1.0];

    RefitArrheniusBTrials = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1];
    RefitArrheniusLnAScaleTrials = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];

    MinimisationFunctions = ['TNC', 'SLSQP', 'L-BFGS-B'];

    DefaultToleranceFitCurves = 1.0e-8;
    DefaultToleranceArrheniusRefit = 1.0e-4;

    DefaultMaxCycles = 10000;

    FitTypesDescriptions = {
        'Independent' : "Independent",
        'SimultaneousN' : "Simultaneous $n$",
        'FixedN' : "$n$ = 1"
        };


# ==============
# Static Methods
# ==============

def JMAKEquation(t, alpha0, alphaF, k, n):
    return alphaF + (alpha0 - alphaF) * np.exp(-1.0 * k * t ** n);
