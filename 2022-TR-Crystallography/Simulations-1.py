# Simulations-1.py


CycleTimes = [170.0, 108.0, 35.0, 22.0, 14.0]

TempMin = 250.0
TempMax = 300.0
TempStep = 0.5

FExcStep = 0.01


import csv
import math
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText

from KineticAnalysis.NumericalSimulator import NumericalSimulator


GasConstant = 8.3145

KExc = 4.3529011372791859e-02
DecEA = 7.4280524105418579e+01
DecLnA = 3.0365813485244846e+01


PPEqThreshold = 1.0e-4


def ezip(*iterables):
    return enumerate(zip(*iterables))

def _SimulateCycle(t_exc, t_cyc, exc_k, dec_k):
    # Create a NumericalSimulator with the supplied excitation and decay rates.
    
    simulator = NumericalSimulator(
        excK = exc_k, excN = 1.0, decK = dec_k, decN = 1.0
        )
    
    # Run repeated pump-probe cycle until the behaviour stabilises.
    
    a_init = 0.0
    
    while True:
        simulator.InitialiseTrajectory(a_init)
        
        if t_exc > 0.0:
            simulator.SetExcitation(True)
            simulator.RunTrajectory(t_exc)
        
        if t_exc < t_cyc:
            simulator.SetExcitation(False)
            simulator.RunTrajectory(t_cyc - t_exc)
        
        sim_t, sim_a = simulator.GetTrajectory()
        
        if math.fabs(sim_a[-1] - a_init) < PPEqThreshold:
            break
        
        a_init = sim_a[-1]
    
    sim_t, sim_a = simulator.GetTrajectory()
    
    # Return minimum/maximum and range of alpha.
    
    a_min = sim_a.min()
    a_max = sim_a.max()
    
    return (a_min, a_max, a_max - a_min)

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
    # Parameter space to sweep.
    
    temps = np.arange(
        TempMin, TempMax + TempStep / 10.0, TempStep
        )
    
    f_exc = np.arange(
        0.0, 1.0 + FExcStep / 10.0, FExcStep
        )
    
    # Perform simulations and store results to CSV files.
    
    for t_cyc in CycleTimes:
        output_file = r"Simulations-1-{0:.0f}s.csv".format(t_cyc)
        
        if not os.path.isfile(output_file):
            data_grid = []
            
            for temp in temps:
                # Work out decay rate from Arrhenius parameters.
                
                dec_k = math.exp(DecLnA) * math.exp(-1.0 * DecEA * 1000.0 / (GasConstant * temp))
                
                data_grid_row = []
                
                for f in f_exc:             
                    t_exc = f * t_cyc
                    
                    a_min, a_max, a_rng = _SimulateCycle(t_exc, t_cyc, KExc, dec_k)
                    
                    print("  -> T = {0:.1f}, f_exc = {1:.2f}, a_min = {2:.3f}, a_max = {3:.3f}, a_rng = {4:.3f}".format(temp, f, a_min, a_max, a_rng))
                    
                    data_grid_row.append(
                        (a_min, a_max, a_rng)
                        )
                
                data_grid.append(data_grid_row)
            
            with open(output_file, 'w', newline = '') as output_writer:
                output_writer_csv = csv.writer(output_writer)
                
                output_writer_csv.writerow(
                    ["T [K]", "f_exc", r"\alpha_min", r"\alpha_max", r"\alpha_rng"]
                    )
                
                for i, temp in enumerate(temps):
                    for j, f in enumerate(f_exc):
                        output_writer_csv.writerow(
                            (temp, f) + data_grid[i][j]
                            )
    
    # Read back CSV files and analyse.
    
    plot_v_rng = [
        (0.0, 0.6  ),
        (0.0, 0.5  ),
        (0.0, 0.25 ),
        (0.0, 0.175),
        (0.0, 0.12 )
        ]
    
    for t_cyc, (v_rng_min, v_rng_max) in zip(CycleTimes, plot_v_rng):
        # Read simulation results from CSV file.
        
        input_file = r"Simulations-1-{0:.0f}s.csv".format(t_cyc)
        
        data_grid = {
            temp : { f : None for f in f_exc }
                for temp in temps
            }
        
        with open(input_file, 'r') as input_reader:
            input_reader_csv = csv.reader(input_reader)
            
            next(input_reader_csv)
            
            for row in input_reader_csv:
                temp, f, a_min, a_max, a_rng = [float(val) for val in row]
                
                assert data_grid[temp][f] is None
                
                data_grid[temp][f] = (a_min, a_max, a_rng)
        
        for v_1 in data_grid.values():
            for v_2 in v_1.values():
                assert v_2 is not None
        
        # Convert to 3 x 2D NumPy arrays.
        
        data_grid = [
            [data_grid[temp][f] for temp in temps]
                for f in f_exc
            ]
        
        data_grid = np.array(data_grid, dtype = np.float64)
        
        a_min = data_grid[:, :, 0]
        a_max = data_grid[:, :, 1]
        a_rng = data_grid[:, :, 2]
        
        # Determine optimum temperature and excitation fraction.
        
        header = "t_cyc = {0:.0f} s".format(t_cyc)
        
        print(header)
        print('-' * len(header))
        
        iy, ix = np.unravel_index(
            np.argmax(a_rng), a_rng.shape
            )
        
        print("  -> T = {0:.1f} w/ f_exc = {1:.3f}/t_exc = {2: >4.1f}: a_min = {3:.3f}, a_max = {4:.3f}, a_rng = {5:.3f}".format(temps[ix], f_exc[iy], t_cyc * f_exc[iy], a_min[iy, ix], a_max[iy, ix], a_rng[iy, ix]))

        a_rng_tmp = np.copy(a_rng)
        a_rng_tmp[a_min > 1.0e-4] = 0.0
        
        iy, ix = np.unravel_index(
            np.argmax(a_rng_tmp), a_rng_tmp.shape
            )
        
        print("  -> T = {0:.1f} w/ f_exc = {1:.3f}/t_exc = {2: >4.1f}: a_min = {3:.3f}, a_max = {4:.3f}, a_rng = {5:.3f}".format(temps[ix], f_exc[iy], t_cyc * f_exc[iy], a_min[iy, ix], a_max[iy, ix], a_rng[iy, ix]))
        
        print("")
        
        # Plot results.
        
        plt.figure(
            figsize = (8.6 / 2.54, 16.0 / 2.54)
            )
        
        grid_spec = GridSpec(3, 7)
        
        subplot_axes = [
            plt.subplot(grid_spec[i, :-1])
                for i in range(3)
            ]
        
        colourbar_axes = [
            plt.subplot(grid_spec[i, -1])
                for i in range(3)
            ]
        
        for i, (grid, ax, cax) in ezip([a_min, a_max, a_rng], subplot_axes, colourbar_axes):
            ret = ax.pcolormesh(
                temps, f_exc, grid, cmap = 'plasma',
                vmin = v_rng_min if i == 2 else 0.0,
                vmax = v_rng_max if i == 2 else 1.0
                )
            
            plt.colorbar(
                ret, ax = ax, cax = cax
                )
        
        labels = [
            r"$\alpha_\mathrm{min}$",
            r"$\alpha_\mathrm{max}$",
            r"$\Delta \alpha$"
            ]
        
        for cax, label in zip(colourbar_axes, labels):
            cax.set_ylabel(label)
        
        for axes in subplot_axes[:-1]:
            axes.set_xticklabels([])
        
        subplot_axes[-1].set_xlabel(r"$T$ [K]")
        
        for axes in subplot_axes:
            axes.set_ylabel(r"$f_\mathrm{exc}$")

        for i, axes in enumerate(subplot_axes):
            anchored_text = AnchoredText(
                r"({0})".format(chr(97 + i)), loc = 1, frameon = True
                )
            
            anchored_text.patch.set_facecolor((1.0, 1.0, 1.0, 0.5))
            anchored_text.patch.set_edgecolor('r')
            
            axes.add_artist(anchored_text)
                
        plt.tight_layout()
        
        plt.savefig(
            r"Simulations-1-{0:.0f}s.png".format(t_cyc), format = 'png', dpi = 300
            )
                
        #plt.close()
