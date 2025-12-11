# Import packages
# General utilities
import os
import random
from datetime import datetime
import subprocess
# Building
from swiftpol import build
from swiftpol import parameterize
from rdkit import Chem
# Parameterization and export
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.interchange import Interchange
from openff.interchange.components._packmol import UNIT_CUBE

# Build polymer system with target PDI, composition and blockiness
sys = build.polymer_system_from_PDI(monomer_list=['O[C@H](C)C(=O)O[I]', 'OCC(=O)O[I]'], 
                               reaction='[C:1][O:2][H:3].[I:4][O:5][C:6]>>[C:1][O:2][C:6].[H:3][O:5][I:4]',
                               length_target=100, 
                               terminals='ester', 
                               num_chains=25,
                               PDI_target=5.0,
                               perc_A_target=74.3,
                               stereoisomerism_input=['A', 0.5, 'O[C@@H](C)C(=O)O[I]'], 
                               blockiness_target=[0.40, 'B'], 
                               copolymer=True,
                               acceptance=20)

# Function to generate random coordinates for a molecule
def generate_random_coordinates(mol):
    """
    Assign random 3D coordinates to all atoms in the molecule.
    """
    num_atoms = mol.GetNumAtoms()
    conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        # Generate random x, y, z coordinates
        x, y, z = random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)
        conf.SetAtomPosition(i, (x, y, z))
    mol.RemoveAllConformers()  # Clear existing conformers from RDKit molecule
    mol.AddConformer(conf, assignId=True)

# Assign random coordinates to each molecule in the system
for mol in sys.chain_rdkit:
    generate_random_coordinates(mol)

sys.chains = [Molecule.from_rdkit(m) for m in sys.chain_rdkit]
# Charge the polymer chains with selected charge model
sys.charge_system('NAGL')

# Create output folder
now = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
dir_name = './outputs/glass_transition/X/output_' + str(now)
os.makedirs(dir_name, exist_ok=True)
# Enter output folder
os.chdir(dir_name)
# Export metadata for system
sys.export_to_csv('metadata.csv')

# Check that PDB residue info is assigned (essential for Polyply)
for atom in sys.chain_rdkit[0].GetAtoms():
    assert atom.GetPDBResidueInfo() is not None

for chain in sys.chains:
    assert chain.name is not None

# Generate residual monomer monomer molecules (if applicable)
molecules, number_of_copies, residual_monomer_actual, residual_oligomer_actual = sys.calculate_residuals(residual_monomer=1.94, 
                                                                                                         residual_oligomer=0)
print(f'residual monomer = {residual_monomer_actual}')

# Generate conformers and unique atom names for residual molecules
molecules[0].generate_conformers(n_conformers=1)
molecules[1].generate_conformers(n_conformers=1)
molecules[0].generate_unique_atom_names()
molecules[1].generate_unique_atom_names()
# Charge residual molecules with selected charge model
parameterize.charge_openff_polymer(molecules[0], 'NAGL')
parameterize.charge_openff_polymer(molecules[1], 'NAGL')

# Create an OpenFF topology containing chains + residual molecules.
topology_off = Topology.from_molecules(sys.chains)+Topology.from_molecules([molecules[0]]*number_of_copies[0])+Topology.from_molecules([molecules[1]]*number_of_copies[1]) #+Topology.from_molecules([risp])

interchange = Interchange.from_smirnoff(topology = topology_off,
                                        force_field=ForceField("openff-2.0.0.offxml"), 
                                        charge_from_molecules=list(set(sys.chains))+[molecules[0], molecules[1]], # Include charges generated using ML model
                                        box = 5*UNIT_CUBE* unit.nanometer # Dummy box size required for GROMACS export
)
interchange.to_top('swiftpol_gromacs.top') # Export GROMACS topology file - polyply input
# Generate initial coordinates using Polyply - subprocess run
subprocess.run(['polyply gen_coords -p swiftpol_gromacs.top -name test -dens 1300 -o swiftpol_gromacs.gro'], check=True, shell=True)
# Energy minimization of initial structure using GROMACS. min.mdp file available on polyply github
subprocess.run(['gmx', 'grompp', '-f', 'min.mdp', '-c', 'swiftpol_gromacs.gro', '-p', 'swiftpol_gromacs.top', '-o', 'em.tpr', '-maxwarn', '10'], check=True)
subprocess.run(['gmx', 'mdrun', '-deffnm', 'em'], check=True)

# Output files:
# swiftpol_gromacs.top - GROMACS topology file
# em.gro - energy minimized coordinates file at 1.3kg/m^3 density