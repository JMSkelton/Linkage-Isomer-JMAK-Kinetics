# KineticAnalyis/Utilities.py


# =======
# Imports
# =======

import csv;
import math;
import sys;

import numpy as np;


# =============
# I/O Functions
# =============

def OpenForCSVWriter(filePath):
    if sys.platform.startswith("win"):
        if sys.version_info.major == 3:
            return open(filePath, 'w', newline = '');
        else:
            print("WARNING: _OpenForCSVWriter(): CSV files output from Python < 3 on Windows platforms may have blank lines between rows.");

    return open(filePath, 'w');

def ReadKineticCurves(filePath, timeScale = 1.0):
    temperatures, curves = [], [];

    with open(filePath, 'r') as inputReader:
        inputReaderCSV = csv.reader(inputReader);

        # Skip header row.

        next(inputReaderCSV);

        # Read data.

        currentTemperature = None;
        currentTimes, currentAlphaMS = [], [];

        for row in inputReaderCSV:
            temperature, time, alphaMS = [float(item) for item in row];

            if currentTemperature != None and temperature != currentTemperature:
                temperatures.append(currentTemperature);

                curves.append(
                    (np.array(currentTimes, dtype = np.float64), np.array(currentAlphaMS, dtype = np.float64))
                    );

                currentTimes, currentAlphaMS = [], [];

            currentTemperature = temperature;

            currentTimes.append(time);
            currentAlphaMS.append(alphaMS);

        if len(currentTimes) != None:
            temperatures.append(currentTemperature);

            curves.append(
                (np.array(currentTimes, dtype = np.float64), np.array(currentAlphaMS, dtype = np.float64))
                );

    # If timeScale is other than 1.0, scale the times.

    if timeScale != 1.0:
        for i, (times, alphaMS) in enumerate(curves):
            curves[i] = (times * timeScale, alphaMS);

    # Return data.

    return (temperatures, curves);

def ReadSSOccupations(filePath):
    temperatures, ssAlphaMS = [], [];

    with open(filePath, 'r') as inputReader:
        inputReaderCSV = csv.reader(inputReader);

        # Skip header row.

        next(inputReaderCSV);

        # Read data.

        for row in inputReaderCSV:
            t, alphaMS = [float(item) for item in row];

            temperatures.append(t);
            ssAlphaMS.append(alphaMS);

    # Return data.

    return (temperatures, ssAlphaMS);


# ==================
# Plotting Functions
# ==================

def HSBColourToRGB(h, s, b):
    h = h % 360.0;

    tempC = s * b;
    tempMin = b - tempC;

    tempHPrime = h / 60.0;
    tempX = tempC * (1.0 - math.fabs((tempHPrime % 2.0) - 1.0));

    r, g, b = 0.0, 0.0, 0.0;

    if tempHPrime < 1.0:
        r = tempC;
        g = tempX;
        b = 0;
    elif tempHPrime < 2.0:
        r = tempX;
        g = tempC;
        b = 0;
    elif tempHPrime < 3.0:
        r = 0;
        g = tempC;
        b = tempX;
    elif tempHPrime < 4.0:
        r = 0;
        g = tempX;
        b = tempC;
    elif tempHPrime < 5.0:
        r = tempX;
        g = 0;
        b = tempC;
    else:
        r = tempC;
        g = 0;
        b = tempX;

    return (r + tempMin, g + tempMin, b + tempMin);
