# Imports
import os
import time
import openmm
import parmed as pmd
from openmm.app import Simulation
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picoseconds, nanometers
from openmm.app.simulation import Simulation

folder = os.environ['FOLDER'] # Get folder name from environment variable
os.chdir(folder)

# Load GROMACS topology and coordinate files
topology_file = "./swiftpol_gromacs.top" # GROMACS topology file
coordinate_file = "./em.gro" # GROMACS coordinate file (from polyply)
structure = pmd.load_file(topology_file, xyz=coordinate_file) # Load structure using ParmEd

# Create OpenMM system
system = structure.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=0.9*nanometers)

# Extract topology and positions
topology = structure.topology
positions = structure.positions

# Define an integrator
integrator = LangevinIntegrator(300*kelvin, # Temperature = 300K
                                1/picoseconds, # Friction coefficient = 1/ps
                                0.001*picoseconds) # Time step = 1 fs

# Set up the simulation
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

# Logging options.
trj_freq = 1000000  # number of steps per written trajectory frame
data_freq = 1000000  # number of steps per written simulation statistics
num_steps = 20000000  # number of integration steps to run - 20ns

simulation.reporters.append(openmm.app.PDBReporter("trajectory_NVT_pdb.pdb", trj_freq))
simulation.reporters.append(openmm.app.DCDReporter("trajectory_NVT_dcd.dcd", trj_freq))
assert simulation.system.usesPeriodicBoundaryConditions() # Ensure periodic boundary conditions are used
# Energy minimization before starting NVT
state = simulation.context.getState(getEnergy=True)
print(state.getPotentialEnergy()) # Print initial potential energy
simulation.minimizeEnergy() # Minimize energy
print(state.getPotentialEnergy()) # Print potential energy after minimization

state_data_reporter = openmm.app.StateDataReporter(
    "data_NVT.csv",
    reportInterval=data_freq,
    step = True,             # writes the step number to each line
    time = True,             # writes the time (in ps)
    potentialEnergy = True,  # writes potential energy of the system (KJ/mole)
    kineticEnergy = True,    # writes the kinetic energy of the system (KJ/mole)
    totalEnergy = True,      # writes the total energy of the system (KJ/mole)
    temperature = True,      # writes the temperature (in K)
    volume = True,           # writes the volume (in nm^3)
    density = True)         # writes the density (in g/mL)

# Append state reporters
simulation.reporters.append(state_data_reporter)

#Simulation
print("Starting equilibration...")
start = time.process_time()

# Run the simulation
simulation.step(num_steps)

# save the equilibration results to file
simulation.saveState('eq.state')
simulation.saveCheckpoint('eq.chk')

end = time.process_time()
print(f"Performed NVT step, temp = {temperature}, time = {(num_steps) / 1000000} ns. Elapsed time {end - start} seconds")

#Save system for reinitialization
system = simulation.context.getSystem()
with open('equil.xml', 'w') as output:
    output.write(openmm.XmlSerializer.serialize(system))

