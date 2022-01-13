# TimeResolved.py


import csv
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from scipy.interpolate import interp1d

from FitLib.Fitter import Fitter
from FitLib.Function import CreateFunction

from KineticAnalysis.NumericalSimulator import NumericalSimulator


GasConstant = 8.3145

KExcInit = 4.3529011372791859e-02
DecEA = 7.4280524105418579e+01
DecLnA = 3.0365813485244846e+01

PPEqThreshold = 1.0e-4


def _InitialiseMatplotlib():
    font_size = 8
    line_width = 0.5

    # Fix for matplotlb.font_manager defaulting to bold variant of Times New Roman on some systems -- adapted from https://github.com/matplotlib/matplotlib/issues/5574.

    #try:
    #    del mpl.font_manager.weight_dict['roman']
    #    mpl.font_manager._rebuild()
    #except KeyError:
    #    pass;

    mpl.rc('font', **{ 'family' : 'serif', 'size' : font_size, 'serif' : 'Times New Roman' })
    mpl.rc('mathtext', **{ 'fontset' : 'custom', 'rm' : 'Times New Roman', 'it' : 'Times New Roman:italic', 'bf' : 'Times New Roman:bold' })

    mpl.rc('axes', **{ 'labelsize' : font_size, 'linewidth' : line_width })
    mpl.rc('lines', **{ 'linewidth' : line_width, 'markeredgewidth' : line_width })

    tickParams = { 'major.width' : line_width, 'minor.width' : line_width, 'direction' : 'in' }

    mpl.rc('xtick', **tickParams )
    mpl.rc('ytick', **tickParams )
    
    mpl.rc('xtick', **{ 'top' : True })
    mpl.rc('ytick', **{ 'right' : True })
    
    mpl.rc('patch', **{ 'linewidth' : line_width })


if __name__ == "__main__":
    # Read input data.
    
    input_data_sets = { }
    
    with open(r"Data/TimeResolved.csv", 'r') as input_reader:
        input_reader_csv = csv.reader(input_reader)
        
        for row in input_reader_csv:
            if "Label" in row[0]: # row[0] == "Label":
                label = row[1]
                
                temp = float(
                    next(input_reader_csv)[1]
                    )
                
                t_exc = float(
                    next(input_reader_csv)[1]
                    )
                
                t_cyc = float(
                    next(input_reader_csv)[1]
                    )
                
                next(input_reader_csv)
                
                t_vals, a_vals = [], []
                
                for row in input_reader_csv:                    
                    if row[0] == '':
                        break
                    
                    t_vals.append(float(row[0]))
                    a_vals.append(float(row[1]))
                
                t_vals = np.array(t_vals, dtype = np.float64)
                a_vals = np.array(a_vals, dtype = np.float64)
                
                input_data_sets[label] = (temp, t_exc, t_cyc, t_vals, a_vals)
    
    # Initialise Matplotlib.
    
    _InitialiseMatplotlib()
    
    # Define general fit functions.
    
    def _SimulateCycle(t_exc, t_cyc, alpha_bg, exc_k, dec_k):
        # Create a NumericalSimulator with the supplied excitation and decay rates.
        
        simulator = NumericalSimulator(
            excK = exc_k, excN = 1.0, decK = dec_k, decN = 1.0
            )
        
        # Run the pump probe cycle.
        
        a_init = 0.0
        
        while True:
            simulator.InitialiseTrajectory(a_init)
            
            simulator.SetExcitation(True)
            simulator.RunTrajectory(t_exc)
            
            simulator.SetExcitation(False)
            simulator.RunTrajectory(t_cyc - t_exc)
            
            sim_t, sim_a = simulator.GetTrajectory()
            
            if math.fabs(sim_a[-1] - a_init) < PPEqThreshold:
                break
            
            a_init = sim_a[-1]
        
        sim_t, sim_a = simulator.GetTrajectory()
        
        # Add on background population.
        
        sim_a += alpha_bg
        
        # Return trajectory.
        
        return (sim_t, sim_a)
    
    def _SimulateCycle2(t_vals, t_exc, t_cyc, alpha_bg, exc_k, dec_k):
        # Generate simulated trajectory.
        
        sim_t, sim_a = _SimulateCycle(t_exc, t_cyc, alpha_bg, exc_k, dec_k)
        
        # Interpolate trajectory to obtain data at supplied times.
        
        interp_func = interp1d(sim_t, sim_a, kind = 'cubic')
        
        return interp_func(t_vals)
    
    # Process datasets.
    
    for label, (temp, t_exc, t_cyc, t_vals, a_vals) in input_data_sets.items():
        header = "Processing: {0} (T = {1:.1f} K, t_exc = {2:.1f} s, t_cyc = {3:.1f} s)".format(label, temp, t_exc, t_cyc)
        
        print(header)
        print('-' * len(header))
        
        # Define fit function.
        
        fit_iter_count = None
        
        def _FitFunc(t_vals, alpha_bg, exc_k, dec_k):
            res = _SimulateCycle2(t_vals, t_exc, t_cyc, alpha_bg, exc_k, dec_k)
            
            global fit_iter_count
            
            if fit_iter_count is not None:
                fit_iter_count += 1
                
                print("_FitFunc(): Iteration {0:0>4}: alpha_bg = {1:.3f}, k_exc = {2:.3e}, k_dec = {3:.3e}".format(fit_iter_count, alpha_bg, exc_k, dec_k))
            
            return res
        
        # HACK.
        
        t_vals, a_vals = t_vals[1:], a_vals[1:]
        
        # Estimate initial decay rate from temperature and Arrhenius parameters
        
        dec_k_init = math.exp((-1.0 * (DecEA * 1000.0) / (GasConstant * temp)) + DecLnA)
        
        # Estimate initial background from data.
        
        alpha_bg_init = a_vals[-1]

        # Refine k_exc with estimated background and k_dec.
        
        fit_func = CreateFunction(
            _FitFunc, [alpha_bg_init, KExcInit, dec_k_init],
            p_fit = [False, True, False], p_bounds = [(0.0, 1.0), (1.0e-8, None), (1.0e-8, None)]
            )
        
        fit_iter_count = 0
        
        fitter = Fitter(
            (t_vals, a_vals), fit_func
            )
        
        rms_fit_init = fitter.Fit()
        
        print("")
        
        _, exc_k_init, _ = [
            p.Value for p in fitter.GetParamsList()
            ]
                
        # Refine alpha_bg, k_exc and k_dec together.
        
        fit_func = CreateFunction(
            _FitFunc, [alpha_bg_init, exc_k_init, dec_k_init],
            p_fit = [True, True, True], p_bounds = [(0.0, 1.0), (1.0e-8, None), (1.0e-8, None)]
            )
        
        fit_iter_count = 0
        
        fitter = Fitter(
            (t_vals, a_vals), fit_func
            )
        
        rms_fit = fitter.Fit()
        
        print("")
        
        alpha_bg_fit, exc_k_fit, dec_k_fit = [
            p.Value for p in fitter.GetParamsList()
            ]
        
        # Plot.
        
        plt.figure(
            figsize = (8.6 / 2.54, 7.0 / 2.54)
            )
        
        plt.scatter(
            t_vals, a_vals, label = r"Expt.", marker = '^', s = 25.0, facecolor = 'none', edgecolor = 'k'
            )
        
        sim_t, sim_a = _SimulateCycle(t_exc, t_cyc, alpha_bg_fit, exc_k_fit, dec_k_fit)
        
        plt.plot(
            sim_t, sim_a, label = r"Sim. Fit", color = 'r'
            )
        
        legend = plt.legend(
            loc = 'upper right', frameon = True
            )
        
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('k')
        
        plt.xlabel(r"$t$ (s)")
        plt.ylabel(r"$\alpha (t)$")
        
        plt.xlim(0.0, t_cyc)
        
        y_ticks, _ = plt.yticks()
        
        inc = y_ticks[1] - y_ticks[0]

        y_min = inc * math.floor(
            min(a_vals.min(), sim_a.min()) / inc
            )
        
        y_max = inc * math.ceil(
            (1.2 * inc) + (max(a_vals.max(), sim_a.max()) / inc)
            )
        
        r_1 = Rectangle(
            (0.0, 0.0), t_exc, y_max, facecolor = 'orange', edgecolor = 'none', alpha = 0.2
            )

        r_2 = Rectangle(
            (t_exc, 0.0), t_cyc - t_exc, y_max, facecolor = 'b', edgecolor = 'none', alpha = 0.2
            )
        
        plt.gca().add_artist(r_1)
        plt.gca().add_artist(r_2)
        
        plt.ylim(y_min, y_max)
        
        y_step = inc
        
        if (y_max - y_min) / y_step < 6:
            y_step /= 2.0
        
        plt.yticks(
            np.arange(y_min, y_max + 1.0e-5, y_step)
            )
        
        plt.tight_layout()
        
        plt.savefig(
            r"TimeResolved-{0}.png".format(label), format = 'png', dpi = 300
            )
        
        # Save fit parameters to a text file.
        
        param_file = r"TimeResolved-{0}.yaml".format(label)
        
        with open(param_file, 'w') as output_writer:
            output_writer.write("label: {0}\n".format(label))
            output_writer.write("temp: {0}\n".format(temp))
            output_writer.write("t_exc: {0}\n".format(t_exc))
            output_writer.write("t_cyc: {0}\n".format(t_cyc))
            
            t_vals_str = ", ".join("{0}".format(t) for t in t_vals)
            a_vals_str = ", ".join("{0}".format(a) for a in a_vals)
            
            output_writer.write("data_t: [ {0} ]\n".format(t_vals_str))
            output_writer.write("data_a: [ {0} ]\n".format(a_vals_str))
            
            output_writer.write("alpha_bg: {0}\n".format(alpha_bg_fit))
            output_writer.write("k_exc: {0}\n".format(exc_k_fit))
            output_writer.write("k_dec: {0}\n".format(dec_k_fit))
            output_writer.write("rms: {0}\n".format(rms_fit))
