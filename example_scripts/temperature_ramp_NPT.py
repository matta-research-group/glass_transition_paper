import os
import numpy as np
import openmm

from openmm import MonteCarloBarostat
from openmm.app.simulation import Simulation
import parmed as pmd
from openmm.unit import nanometers
from openmm import unit 

folder = os.environ['FOLDER']
os.chdir(folder)

trj_freq = 1000000  # number of steps per written trajectory frame
data_freq = 1000000  # number of steps per written simulation statistics

# Load GROMACS topology and coordinate files
topology_file = "./swiftpol_gromacs.top"
coordinate_file = "./em.gro"
structure = pmd.load_file(topology_file, xyz=coordinate_file)

sys = structure.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=0.9*nanometers)
top = structure.topology

# Load equilibrated system
with open('equil.xml') as input:
    system = openmm.XmlSerializer.deserialize(input.read())

# Integration options
time_step = 1 * unit.femtoseconds  # simulation timestep
temperature = 500 * unit.kelvin  # simulation temperature
friction = 1 / unit.picosecond  # friction constant

# Initialize integrator, barostat, and simulation
integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
barostat = system.addForce(MonteCarloBarostat(1, 500))
simulation = Simulation(top, system, integrator, state = 'eq.state') # Load positions from equilibration

assert simulation.system.usesPeriodicBoundaryConditions() == True # Ensure periodic boundary conditions are used
simulation.minimizeEnergy() # Minimize energy
# Initialize reporters
dcd_reporter = openmm.app.DCDReporter("trajectory_NPT_dcd_500_equil.dcd", trj_freq)
state_data_reporter = openmm.app.StateDataReporter(
    "data_NPT_tempramp_500_equil.csv",
    reportInterval=data_freq,
    step = True,             # writes the step number to each line
    time = True,             # writes the time (in ps)
    potentialEnergy = True,  # writes potential energy of the system (KJ/mole)
    kineticEnergy = True,    # writes the kinetic energy of the system (KJ/mole)
    totalEnergy = True,      # writes the total energy of the system (KJ/mole)
    temperature = True,      # writes the temperature (in K)
    volume = True,           # writes the volume (in nm^3)
    density = True)          # writes the density (in g/mL)

checkpoint_reporter = openmm.app.checkpointreporter.CheckpointReporter("checkpoint_prod_tempramp_500_equil.chk", 1000, writeState=True)
# Append reporters to simulation
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(state_data_reporter)
simulation.reporters.append(checkpoint_reporter)

# NPT
# Length of the simulation.
num_steps = 60000000  # number of integration steps to run - 60ns
print(f"Running simulation. Temp = {simulation.context.getIntegrator().getTemperature()}, "
      f"pressure = {simulation.context.getParameter(MonteCarloBarostat.Pressure())}, "
      f"Time = {(num_steps * 1) / 1000000} ns")
simulation.step(num_steps)
print(f"Ran simulation. Temp = {temperature}, "
      f"pressure = {simulation.context.getParameter(MonteCarloBarostat.Pressure())}, "
      f"Time = {(num_steps * 1) / 1000000} ns")

# Cooling ramp
start_temp = 500
min_temp = 200
quench_rate = 0.5 #K/ns
cool_ramp = np.arange(min_temp, start_temp + quench_rate, quench_rate).tolist()
cool_ramp.reverse()

for i in cool_ramp: #change range to change temp ramp speed
    simulation.reporters.clear()
    temperature = i*openmm.unit.kelvin
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), temperature)
    integrator.setTemperature(temperature)
    # Create new reporters with temperature in the title
    dcd_reporter = openmm.app.DCDReporter(f"trajectory_NPT_dcd_tempramp_{str(i).replace('.', '-')}K.dcd", trj_freq)
    state_data_reporter = openmm.app.StateDataReporter(
        f"data_NPT_tempramp_{str(i).replace('.', '-')}K.csv",
        reportInterval=data_freq,
        step=True,             # writes the step number to each line
        time=True,             # writes the time (in ps)
        potentialEnergy=True,  # writes potential energy of the system (KJ/mole)
        kineticEnergy=True,    # writes the kinetic energy of the system (KJ/mole)
        totalEnergy=True,      # writes the total energy of the system (KJ/mole)
        temperature=True,      # writes the temperature (in K)
        volume=True,           # writes the volume (in nm^3)
        density=True)          # writes the density (in g/mL)

    checkpoint_reporter = openmm.app.checkpointreporter.CheckpointReporter(f"checkpoint_prod_tempramp_{str(i).replace('.', '-')}K.chk", 10000000, writeState=True)

    # Append the new reporters to the simulation
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_data_reporter)
    simulation.reporters.append(checkpoint_reporter)
    num_steps = 1000000
    simulation.step(num_steps)
    # 10ns hold for each 10K drop
    if i % 10 == 0:
        print(f"Running simulation. Temp = {simulation.context.getIntegrator().getTemperature()}, "
              f"pressure = {simulation.context.getParameter(MonteCarloBarostat.Pressure())}, "
              f"Time = {(num_steps) / 1000000} ns")
        num_steps = 10000000
        simulation.step(num_steps)
        print(f"Ran simulation. Temp = {simulation.context.getIntegrator().getTemperature()}, "
              f"pressure = {simulation.context.getParameter(MonteCarloBarostat.Pressure())}, "
              f"Time = {(num_steps) / 1000000} ns")



