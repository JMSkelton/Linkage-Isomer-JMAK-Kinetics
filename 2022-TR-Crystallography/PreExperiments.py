# PreExperiments.py


import csv
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import AnchoredText

from scipy.interpolate import interp1d

from FitLib.Convenience import SweepInitialParameters
from FitLib.Fitter import Fitter
from FitLib.Function import CreateFunction
from FitLib.FunctionsLibrary.Kinetics import JMAK

from KineticAnalysis.NumericalSimulator import NumericalSimulator


GasConstant = 8.3145

JMAKKSweep = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]


def _ReadKineticCurves(file_path):
    with open(file_path, 'r') as input_reader:
        input_reader_csv = csv.reader(input_reader)
        
        temps = [
            float(item.replace("T = ", '').replace("K", ''))
                for item in next(input_reader_csv)
                    if item != ""
            ]
        
        next(input_reader_csv)
        
        data_cols = [
            [] for _ in range(len(temps) * 2)
            ]
        
        for row in input_reader_csv:
            for i, val in enumerate(row):
                data_cols[i].append(float(val))
        
        data_cols = [
            np.array(col, dtype = np.float64)
                for col in data_cols
            ]
        
        if len(temps) == 1:
            return (temps[0], tuple(data_cols))
        else:
            return {
                temp : tuple(data_cols[2 * i:2 * (i + 1)])
                    for i, temp in enumerate(temps)
                }

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

def _HSBColourToRGB(h, s, b):
    h %= 360.0

    temp_c = s * b
    temp_min = b - temp_c

    temp_h_prime = h / 60.0
    temp_x = temp_c * (1.0 - math.fabs((temp_h_prime % 2.0) - 1.0))

    r, g, b = 0.0, 0.0, 0.0

    if temp_h_prime < 1.0:
        r = temp_c
        g = temp_x
        b = 0
    elif temp_h_prime < 2.0:
        r = temp_x
        g = temp_c
        b = 0
    elif temp_h_prime < 3.0:
        r = 0
        g = temp_c
        b = temp_x
    elif temp_h_prime < 4.0:
        r = 0
        g = temp_x
        b = temp_c
    elif temp_h_prime < 5.0:
        r = temp_x
        g = 0
        b = temp_c
    else:
        r = temp_c
        g = 0
        b = temp_x

    return (r + temp_min, g + temp_min, b + temp_min)


if __name__ == "__main__":
    # Initialise Matplotlib.
    
    _InitialiseMatplotlib()
    
    # ----------
    # Decay Data
    # ----------
    
    # Read decay curves.
    
    dec_data_sets = _ReadKineticCurves(r"Data/Pre-Decay.csv")
    
    dec_temps = sorted(
        dec_data_sets.keys()
        )
    
    # Fit decay curves to JMAK function w/ n = 1.
    
    for temp, (t, a_t) in dec_data_sets.items():
        p_opt, fit_results = SweepInitialParameters(
            (t, a_t), JMAK, [a_t[0], a_t[-1], JMAKKSweep, 1.0], ["a_0", "a_inf", "k", "n"],
            param_fit_flags = [True, True, True, False],
            param_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, None), None],
            print_status = False, tol = 1e-8
            )
        
        _, _, rms = fit_results[0]
        
        dec_data_sets[temp] = (
            t, a_t, tuple(p_opt), rms
            )
        
        output_file = r"PreExperiments-1-{0:.1f}K.yaml".format(temp)
        
        with open(output_file, 'w') as output_writer:
            output_writer.write("temp: {0}\n".format(temp))

            a_0, a_inf, k, n = p_opt

            output_writer.write("alpha_0: {0}\n".format(a_0))
            output_writer.write("alpha_inf: {0}\n".format(a_inf))
            output_writer.write("k: {0}\n".format(k))
            output_writer.write("n: {0}\n".format(n))

            output_writer.write("rms: {0}\n".format(rms))
    
    print("Decay curve fits:")
    print("-----------------")
    
    for temp in dec_temps:
        _, _, fit_params, rms = dec_data_sets[temp]
        
        print("T = {0:.1f} K, a_0 = {1:.3f}, a_inf = {2:.3f}, k_dec = {3:.3e}, n_dec = {4:.1f}, rms = {5:.3e}".format(temp, *fit_params, rms))
    
    print("")

    # Fit k(T) to Arrhenius law.

    dec_inv_t, dec_ln_k = [], []
    
    for temp in dec_temps:
        _, _, (_, _, k_dec, _), _ = dec_data_sets[temp]
        
        dec_inv_t.append(1.0 / temp)
        
        dec_ln_k.append(
            math.log(k_dec)
            )
    
    m, c = np.polyfit(
        dec_inv_t, dec_ln_k, deg = 1
        )
    
    dec_e_a = (-1.0 * GasConstant * m) / 1000.0
    dec_ln_a = c

    print("Arrhenius analysis:")
    print("-------------------")
    
    print("E_A   = {0:.2f}".format(dec_e_a ))
    print("ln(A) = {0:.2f}".format(dec_ln_a))
    
    print("")
    
    # Plot decay fits.
    
    angle_increment = 150.0 / (dec_temps[-1] - dec_temps[0])
    
    plot_colours = [
        _HSBColourToRGB(240.0 + (temp - dec_temps[0]) * angle_increment, 1.0, 1.0)
            for temp in dec_temps
        ]
    
    plot_t_min, plot_t_max = 0.0, 36 * 60.0
    
    plot_t_jmak = np.arange(
        plot_t_min, plot_t_max + 1.0e-5, 1.0
        )
    
    plt.figure(
        figsize = (8.6 / 2.54, 7.0 / 2.54)
        )
    
    for temp, colour in zip(dec_temps, plot_colours):
        t, a_t, fit_params, _ = dec_data_sets[temp]
        
        plt.scatter(
            t, a_t, label = r"{0:.1f} K".format(temp),
            marker = '^', s = 25.0, fc = 'none', ec = colour
            )
        
        a_t_fit = JMAK(plot_t_jmak, *fit_params)
        
        plt.plot(
            plot_t_jmak, a_t_fit, color = colour
            )
    
    plt.legend(
        loc = 'upper right', frameon = False, ncol = 2
        )
    
    plt.xlabel(r"$t$ [min]")
    plt.ylabel(r"$\alpha(t)$")
    
    plt.xlim(plot_t_min, plot_t_max)
    plt.ylim(0.0, 1.0)
    
    plt.xticks(
        np.arange(0.0, 36 * 60.0 + 1.0e-5, 6 * 60.0)
        )
    
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos : "{0:.1f}".format(val / 60.0))
        )
    
    plt.tight_layout()
    
    plt.savefig(
        r"PreExperiments-1.png", format = 'png', dpi = 300
        )
    
    #plt.close()
    
    # Plot Arrhenius analysis.
    
    plot_inv_t_min, plot_inv_t_max = 3.75e-3, 4.25e-3
    
    plt.figure(
        figsize = (8.6 / 2.54, 7.0 / 2.54)
        )
    
    plt.scatter(
        dec_inv_t, dec_ln_k, marker = '^', s = 25.0, fc = 'none', ec = 'k'
        )
    
    plt.plot(
        [plot_inv_t_min, plot_inv_t_max], [m * plot_inv_t_min + c, m * plot_inv_t_max + c],
        color = 'k', dashes = (3.0, 1.0)
        )
    
    plt.xlabel(r"$T^{-1}$ [10$^{-3}$ K$^{-1}$]")
    plt.ylabel(r"ln($k_{\mathrm{dec}}$)")
    
    plt.xlim(plot_inv_t_min, plot_inv_t_max)
    plt.ylim(-7.5, -2.5)
    
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos : "{0:.1f}".format(val * 1.0e3))
        )
    
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda val, pos : "{0:.1f}".format(val))
        )
    
    anchored_text = AnchoredText(
        r"$E_{{\mathrm{{A}}}}$ = {0:.1f} kJ mol$^{{-1}}$".format(dec_e_a) + '\n' + r"ln($A$) = {0:.1f}".format(dec_ln_a),
        loc = 1, frameon = False
        )
    
    plt.gca().add_artist(anchored_text)
    
    plt.tight_layout()
    
    plt.savefig(
        r"PreExperiments-2.png", format = 'png', dpi = 300
        )
    
    #plt.close()

    # ---------------
    # Excitation Data
    # ---------------
    
    # Read excitation curve.
    
    exc_temp, (exc_t, exc_a_t) = _ReadKineticCurves(r"Data/Pre-Excitation.csv")
    
    # Fit to JMAK function w/ n = 1.
    
    p_opt, fit_results = SweepInitialParameters(
        (exc_t, exc_a_t), JMAK, [exc_a_t[0], exc_a_t[-1], JMAKKSweep, 1.0], ["a_0", "a_inf", "k", "n"],
        param_fit_flags = [True, True, True, False],
        param_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, None), None],
        print_status = False, tol = 1e-8
        )
    
    exc_fit_params = tuple(p_opt)
    _, _, exc_rms = fit_results[0]
    
    with open(r"PreExperiments-3.yaml", 'w') as output_writer:
        output_writer.write("temp: {0}\n".format(exc_temp))
        
        a_0, a_inf, k, n = p_opt
        
        output_writer.write("alpha_0: {0}\n".format(a_0))
        output_writer.write("alpha_inf: {0}\n".format(a_inf))
        output_writer.write("k: {0}\n".format(k))
        output_writer.write("n: {0}\n".format(n))
        
        output_writer.write("rms: {0}\n".format(exc_rms))

    print("Excitation curve fit:")
    print("---------------------")
    
    print("T = {0:.1f} K, a_0 = {1:.3f}, a_inf = {2:.3f}, k_exc = {3:.3e}, n_exc = {4:.1f}, rms = {5:.3e}".format(exc_temp, *exc_fit_params, exc_rms))
    print("")
    
    # Plot excitation fit.
    
    plot_t_min, plot_t_max = 0.0, 150.0
    
    plot_t_jmak = np.arange(
        plot_t_min, plot_t_max + 1.0e-5, 0.1
        )
    
    plt.figure(
        figsize = (8.6 / 2.54, 7.0 / 2.54)
        )
    
    plt.scatter(
        exc_t, exc_a_t, marker = '^', s = 25.0, fc = 'none', ec = 'b'
        )
        
    a_t_fit = JMAK(plot_t_jmak, *exc_fit_params)
    
    plt.plot(
        plot_t_jmak, a_t_fit, color = 'b'
        )
    
    plt.xlabel(r"Exposure Time $t$ [s]")
    plt.ylabel(r"$\alpha(t)$")
    
    plt.xlim(plot_t_min, plot_t_max)
    plt.ylim(0.0, 1.0)
    
    plt.xticks(
        np.arange(0.0, 150.0 + 1.0e-5, 25.0)
        )
    
    plt.tight_layout()
    
    plt.savefig(
        r"PreExperiments-3.png", format = 'png', dpi = 300
        )
    
    #plt.close()

    # -----------------
    # Steady-State Data
    # -----------------

    # Read steady-state measurements.
    
    t_ss, a_ss = None, None
    
    with open(r"Data/Pre-SteadyState.csv", 'r') as input_reader:
        input_reader_csv = csv.reader(input_reader)
        
        next(input_reader_csv)
        
        t_ss, a_ss = [], []
        
        for row in input_reader_csv:
            t_ss.append(float(row[0]))
            a_ss.append(float(row[1]))
    
    # Refine excitation rate + decay Arrhenius parameters against steady-state measurements.
    
    _, _, exc_k, _ = exc_fit_params
    
    # JMS: For some reason, calling NumericalSimulator.RunTrajectory() isn't doing what it should be.
    # JMS: To work around this, _FitFunc() implements a while loop running the simulator in increments of 1 s.
    
    sim_update_time = 1.0
    sim_break_criterion = 1.0e-4
    
    # JMS: This is a "dirty" hack to display status messages during fitting.
    
    fit_iter_count = None
    
    def _FitFunc(temps, exc_k, dec_e_a, dec_ln_a):
        simulator = NumericalSimulator(
            excN = 1.0, excK = exc_k, decN = 1.0, decArrheniusParams = (dec_e_a, dec_ln_a)
            )
        
        a_ss_sim = []
        
        for temp in temps:
            simulator.SetExcitation(True)
            simulator.SetTemperature(temp)
            
            simulator.InitialiseTrajectory(0.0)
            
            a_sim_last = None
            
            while True:
                simulator.RunTrajectory(runTime = sim_update_time)
                
                _, sim_traj_a = simulator.GetTrajectory()
                
                if a_sim_last is not None and math.fabs(sim_traj_a[-1] - a_sim_last) < sim_break_criterion:
                        break
                    
                a_sim_last = sim_traj_a[-1]
            
            _, a_sim = simulator.GetTrajectory()
            
            a_ss_sim.append(a_sim[-1])
        
        # JMS: Status messages.
        
        global fit_iter_count
        
        if fit_iter_count is not None:
            fit_iter_count += 1
            
            print("_FitFunc(): Iteration {0:0>4}: k_exc = {1:.3e}, dec_e_a = {2:.2f}, dec_ln_a = {3:.2f}".format(fit_iter_count, exc_k, dec_e_a, dec_ln_a))
        
        return a_ss_sim
    
    # Refine the excitation rate.
    
    rms_init = math.sqrt( np.mean( np.subtract( _FitFunc(t_ss, exc_k, dec_e_a, dec_ln_a), a_ss ) ** 2 ) )
    
    print("Refining excitation rate:")
    print("-------------------------")
    
    fit_func = CreateFunction(
        _FitFunc, [exc_k, dec_e_a, dec_ln_a],
        p_fit = [True, False, False], p_bounds = [(1.0e-8, None), (1.0e-8, None), (1.0e-8, None)]
        )
    
    fit_iter_count = 0
    
    fitter = Fitter(
        (t_ss, a_ss), fit_func
        )
    
    rms_fit_1 = fitter.Fit()
    
    print("")
    
    exc_k_fit_1, _, _ = [
        p.Value for p in fitter.GetParamsList()
        ]
    
    print("Refining excitation rate + Arrhenius parameters:")
    print("------------------------------------------------")

    fit_func = CreateFunction(
        _FitFunc, [exc_k_fit_1, dec_e_a, dec_ln_a],
        p_fit = [True, True, True], p_bounds = [(1.0e-8, None), (1.0e-8, None), (1.0e-8, None)]
        )
    
    fit_iter_count = 0
    
    fitter = Fitter(
        (t_ss, a_ss), fit_func
        )
    
    rms_fit_2 = fitter.Fit()

    exc_k_fit_2, dec_e_a_fit_2, dec_ln_a_fit_2 = [
        p.Value for p in fitter.GetParamsList()
        ]
    
    print("")
    
    print("Summary:")
    print("--------")
    
    print("Initial: k_exc = {0:.3e}, dec_e_a = {1:.2f}, dec_ln_a = {2:.2f}, rms = {3:.3e}".format(exc_k      , dec_e_a  , dec_ln_a  , rms_init ))
    print("Fit 1  : k_exc = {0:.3e}, dec_e_a = {1:.2f}, dec_ln_a = {2:.2f}, rms = {3:.3e}".format(exc_k_fit_1, dec_e_a  , dec_ln_a  , rms_fit_1))
    print("Fit 2  : k_exc = {0:.3e}, dec_e_a = {1:.2f}, dec_ln_a = {2:.2f}, rms = {3:.3e}".format(exc_k_fit_2, dec_e_a_fit_2, dec_ln_a_fit_2, rms_fit_2))
    
    print("")
    
    # Run numerical simulations against excitation + decay curves and compare to initial fits.
    
    plot_temp_min, plot_temp_max = 220.0, 320.0
    
    plot_temp_sim = np.arange(
        plot_temp_min, plot_temp_max + 1.0e-5, 1.0
        )
    
    fit_iter_count = None
    
    plot_a_ss_sim_init = _FitFunc(
        plot_temp_sim, exc_k, dec_e_a, dec_ln_a
        )

    plot_a_ss_sim_1 = _FitFunc(
        plot_temp_sim, exc_k_fit_1, dec_e_a, dec_ln_a
        )

    plot_a_ss_sim_2 = _FitFunc(
        plot_temp_sim, exc_k_fit_1, dec_e_a_fit_2, dec_ln_a_fit_2
        )
 
    plt.figure(
        figsize = (8.6 / 2.54, 7.0 / 2.54)
        )
    
    plt.scatter(
        t_ss, a_ss, marker = '^', s = 25.0, fc = 'none', ec = 'k'
        )
    
    plot_params = [
        (plot_a_ss_sim_init, 'b'     , r"Initial"     ),
        (plot_a_ss_sim_1   , 'r'     , r"Refinement 1"),
        (plot_a_ss_sim_2   , 'orange', r"Refinement 2")
        ]
    
    for plot_a_ss_sim, colour, label in plot_params:
        plt.plot(
            plot_temp_sim, plot_a_ss_sim, label = label, color = colour
            )
    
    plt.legend(
        loc = 'upper right', frameon = False
        )
    
    plt.xlabel(r"$T$ [K]")
    plt.ylabel(r"Steady-State $\alpha$")
    
    plt.xlim(plot_temp_min, plot_temp_max)
    plt.ylim(0.0, 1.0)
        
    plt.tight_layout()
    
    plt.savefig(
        r"PreExperiments-4.png", format = 'png', dpi = 300
        )
    
    # ------------
    # Finalisation
    # ------------
    
    # Regenerate each of the excitation and decay curves with the fit parameters.

    print("Simulations:")
    print("------------")

    simulator_1 = NumericalSimulator(
        excN = 1.0, excK = exc_k, decN = 1.0, decArrheniusParams = (dec_e_a, dec_ln_a)
        )
    
    simulator_2 = NumericalSimulator(
        excN = 1.0, excK = exc_k_fit_2, decN = 1.0, decArrheniusParams = (dec_e_a_fit_2, dec_ln_a_fit_2)
        )
    
    simulations = []
    
    for temp in dec_temps:
        simulations.append(
            (temp, ) + dec_data_sets[temp] + (False, 36.0 * 60.0, "Dec-{0:.1f}K".format(temp))
            )
    
    simulations.append(
        (exc_temp, ) + (exc_t, exc_a_t, exc_fit_params, exc_rms) + (True, 160.0, "Exc")
        )
    
    for temp_sim, expt_t, expt_a_t, fit_params, fit_rms, exc_active, t_sim, suffix in simulations:
        a_0 = None
        
        if expt_t[0] == 0.0:
            a_0 = expt_a_t[0]
        else:
            if exc_active:
                a_0 = 0.0
            else:
                a_0 = 1.0
        
        # Simulation with initial parameter set.
        
        simulator_1.SetExcitation(exc_active)
        simulator_1.SetTemperature(temp_sim)
        
        simulator_1.InitialiseTrajectory(a_0)
        simulator_1.RunTrajectory(t_sim)
        
        # Simulation with refined parameter set.
        
        simulator_2.SetExcitation(exc_active)
        simulator_2.SetTemperature(temp_sim)
        
        simulator_2.InitialiseTrajectory(a_0)
        simulator_2.RunTrajectory(t_sim)
        
        # Calculate RMS of simulations.

        interp_func = interp1d(
            *simulator_1.GetTrajectory(), kind = 'cubic'
            )
        
        rms_sim_1 = math.sqrt( np.mean( np.subtract( interp_func(expt_t), expt_a_t ) ** 2 ) )
        
        interp_func = interp1d(
            *simulator_2.GetTrajectory(), kind = 'cubic'
            )
        
        rms_sim_2 = math.sqrt( np.mean( np.subtract( interp_func(expt_t), expt_a_t ) ** 2 ) )
        
        print("Simulation: T = {0:.1f} K, exc = {1: >5}, rms_jmak = {2:.3e}, rms_sim_1 = {3:.3e}, rms_sim_2 = {4:.3e}".format(temp_sim, exc_active, fit_rms, rms_sim_1, rms_sim_2))

        # Plots comparing JMAK fit + simulations with initial and final parameters to experimental data.
        
        plt.figure(
            figsize = (8.6 / 2.54, 7.0 / 2.54)
            )
        
        plt.scatter(
            expt_t, expt_a_t, marker = '^', s = 25.0, fc = 'none', ec = 'k'
            )
        
        jmak_fit_t = np.linspace(0.0, t_sim, 1001)
        jmak_fit_a_t = JMAK(jmak_fit_t, *fit_params)
        
        plt.plot(
            jmak_fit_t, jmak_fit_a_t, label = r"JMAK Fit", color = 'k'
            )

        sim_traj_a, sim_traj_t = simulator_1.GetTrajectory()

        plt.plot(
            sim_traj_a, sim_traj_t, label = r"Sim. (Initial)", color = 'b'
            )

        sim_traj_a, sim_traj_t = simulator_2.GetTrajectory()

        plt.plot(
            sim_traj_a, sim_traj_t, label = r"Sim. (Refined)", color = 'r'
            )
        
        plt.legend(
            loc = 'lower right' if exc_active else 'upper right', frameon = False
            )
        
        # HACK.
        
        if exc_active:
            plt.xlabel(r"Exposure Time $t$ [s]")
        else:
            plt.xlabel(r"$t$ [min]")

        plt.ylabel(r"$\alpha(t)$")
        
        plt.xlim(0.0, t_sim)
        plt.ylim(0.0, 1.0)

        # HACK.

        if not exc_active:
            plt.xticks(
                np.arange(0.0, 36 * 60.0 + 1.0e-5, 6 * 60.0)
                )
            
            plt.gca().xaxis.set_major_formatter(
                FuncFormatter(lambda val, pos : "{0:.1f}".format(val / 60.0))
                )
            
        plt.tight_layout()
        
        plt.savefig(
            r"PreExperiments-5-{0}.png".format(suffix), format = 'png', dpi = 300
            )
        
        #plt.close()
    
    print("")
    
    # Print final values for TR fits to high precision.
    
    print("Initial TR fit parameters:")
    print("--------------------------")
    
    print("exc_k    = {0:.16e}".format(exc_k_fit_2   ))
    print("dec_e_a  = {0:.16e}".format(dec_e_a_fit_2 ))
    print("dec_ln_a = {0:.16e}".format(dec_ln_a_fit_2))
    
    print("")
