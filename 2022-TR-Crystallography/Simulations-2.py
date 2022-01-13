# Simulations-2.py


import glob
import math

import yaml

from KineticAnalysis.NumericalSimulator import NumericalSimulator


PPEqThreshold = 1.0e-4


if __name__ == "__main__":
    # Read fits to time-resolved datasets.
    
    tr_data_sets = { }
    
    for f in glob.glob(r"TimeResolved-*.yaml"):
        with open(f, 'rb') as input_file:
            input_yaml = yaml.load(
                input_file, Loader = yaml.CLoader
                )
            
            tr_data_sets[input_yaml['label']] = (
                input_yaml['t_cyc'],
                input_yaml['t_exc'],
                input_yaml['temp'],
                (input_yaml['data_t'], input_yaml['data_a']),
                (input_yaml['alpha_bg'], input_yaml['k_exc'], input_yaml['k_dec']),
                input_yaml['rms']
                )
    
    # For each dataset, determine the maximum excited-state population given the fitted k_exc/k_dec.
    
    for label, (t_cyc, t_exc, _, _, (_, k_exc, k_dec), _) in tr_data_sets.items():
        simulator = NumericalSimulator(
            excN = 1.0, excK = k_exc, decN = 1.0, decK = k_dec
            )
        
        # 1. Determine the excitation level achieved with the set t_cyc/t_exc.
        
        a_0, a_max = 0.0, 0.0
        
        while True:
            simulator.InitialiseTrajectory(a_0)
            
            simulator.SetExcitation(True)
            simulator.RunTrajectory(t_exc)
            
            _, a_sim = simulator.GetTrajectory()
            
            a_max = a_sim[-1]
            
            simulator.SetExcitation(False)
            simulator.RunTrajectory(t_cyc - t_exc)
            
            _, a_sim = simulator.GetTrajectory()
            
            if math.fabs(a_sim[-1] - a_0) < PPEqThreshold:
                break
            
            a_0 = a_sim[-1]
        
        # 2. Determine the steady-state (maximum) excitation level.
        
        simulator.SetExcitation(True)
        simulator.InitialiseTrajectory(0.0)
        
        a_max_sim = 0.0
        
        while True:
            simulator.RunTrajectory(1.0)
            
            _, a_sim = simulator.GetTrajectory()
            
            if math.fabs(a_sim[-1] - a_max_sim) < PPEqThreshold:
                break
            
            a_max_sim = a_sim[-1]
        
        print("{0}: a_max = {1:.3f}, theoretical = {2:.3f}".format(label, a_max, a_max_sim))
    
    print("")