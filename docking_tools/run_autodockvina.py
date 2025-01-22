#!/usr/bin/env python3

#########################
import mdtraj as md 
import numpy as np 
import os, sys 
from vina import Vina 
import subprocess as sp
import argparse
from openbabel import openbabel 
##########################
# convenience functions: 
from .utils import chk_mkdir, lsdir, get_filename, keyword_strs
from .utils import itermatch, str_csv


# PREPARING AUTODOCK FILES ------------------

def get_res_com(traj_frame, residue):
    """ Returns center of mass in angstroms for a residue in an mdtraj object  """
    com = 10*(md.compute_center_of_mass(traj_frame, select=f'resSeq {residue}'))
    comx = com[0,0].item() # from the provided mdtraj com 
    comy = com[0,1].item()
    comz = com[0,2].item()
    return comx, comy, comz


def get_boxsize(ebox_func, lig_pdbqt):
    """ Returns the float of the boxsize for a given ligand pdbqt and the eBox function """
    bash_cmnd = sp.run(['perl', ebox_func, lig_pdbqt], capture_output=True)
    byte_out = bash_cmnd.stdout
    return float(byte_out.decode())


def mk_lig_pdbqt(lig_file, pdbqt_filepath): 
    """ Prepares the ligand pdbqt using Meeko. """
    sp.run(['mk_prepare_ligand.py', '-i', lig_file, '-o' ,pdbqt_filepath, '--merge_these_atom_types'])
    return pdbqt_filepath


def protein_pdbqt_ADFR(pdb, pdbqt, func): 
    """ Prepare a protein pdbqt from a pdb using the ADFR suite. """ 
    sp.run([func,'-r', pdb, '-A', 'hydrogens', '-o', pdbqt])
    return pdbqt 


def load_protein_traj(traj_file, pdb): 
    """ Returns a trajectory with only the protein. """ 
    traj = md.load(traj_file, top=pdb, stride=1)
    protein_traj = traj.atom_slice(traj.top.select('protein'))
    return protein_traj

def get_frame_pdb(outdir, traj_frame): 
    """ Saves a PDB for a given trajectory frame and returns the file path for the PDB."""
    traj_frame.save_pdb(f"{outdir}/protein.pdb")
    return f'{outdir}/protein.pdb'
	

def get_lowest_energy(vina_outdir): 
    """ Return file, energy, and index of the min energy for my vina outdir. """

    def get_energy(vina_file): 
        """   Returns the docking energy for a vina file.  """
        with open(vina_file, 'r') as open_file: 
            energy_line = open_file.readlines()[1].split()
        return float(energy_line[3])

    file_list = lsdir(vina_outdir)
    en_list = list(map(get_energy, file_list)) 
    
    index = en_list.index(min(en_list))
    return file_list[index], en_list[index], index 


# POST - PROCESSING... ----------------------

def meeko_pdbqt_export(vinafile, sdf): 
    """ Returns the new file path for the exported sdf file, using meeko... """
    sp.run(['mk_export.py', vinafile, '-o', sdf])
    return sdf 

def sdf_to_pdb(sdf, pdb):
    """ Converts sdf files to pdb files using open babel. """ 
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdb")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, sdf) 
    obConversion.WriteFile(mol, pdb) 
    return pdb


# RESORTING INDICES of the docked ligand to match original ligand mol2 file
# to avoid indexing issue for RMSD calculations later 

def ligand_xyz(ligfile, atoms=[' O ', ' S ', ' H ', ' C ', ' N '], filetype:str='mol2'):
    """ Returns coordinates for ligand atoms given atom identities (str list, format = [' C ']). 
    Following supported file types: mol2, pdb, pdbqt, and sdf. """ 

    # Excluding REMARK lines 
    with open(ligfile, 'r') as openfile:
        lines = keyword_strs(openfile.readlines(), keyword="REMARK", exclude=True)

    # Grabbing only the ligand atom lines
    splitter = lambda s : s.replace(" ","").split(",")
    lines = np.array(list(map(splitter, itermatch(map(str_csv, lines), atoms))))

    # For the specific file type, 
    if filetype=="mol2": coor=[2, 3, 4]
    if filetype=='sdf': coor=[0, 1, 2]
    if filetype=='pdb': coor=[6, 7, 8]
    if filetype=='pdbqt': coor=[5, 6, 7]

    # Return the coordinates. 
    return lines[:,coor].astype(float)

def get_sdf_indices(mol2, mk_pdbqt, vina_pdbqt, sdf, 
            atoms=[' O ', ' S ', ' H ', ' C ', ' N ']):
    """ Following the order of conversion (mol2 --> meeko pdbqt --> vina pdbqt --> meeko sdf),  
    gives the indices of the final sdf file so that sdf_atoms[indices] == mol2_atoms.  """
    
    # Loading in coordinates for each file: 
    mol2xyz = ligand_xyz(mol2, atoms, filetype='mol2')
    mk_pdbqtxyz = ligand_xyz(mk_pdbqt, atoms, filetype='pdbqt')
    vina_pdbqtxyz = ligand_xyz(vina_pdbqt, atoms, filetype='pdbqt')
    sdfxyz = ligand_xyz(sdf, atoms, filetype='sdf')
    
    # Getting the indices for the first shuffle (mol2-->mk_pdbqt): 
    ind1 = np.array([np.where((mk_pdbqtxyz== i).all(1))[0][0] for i in mol2xyz])
    
    # Getting the indices for the second shuffle (vina_pdbqt-->mk_sdf: 
    ind2 = np.array([np.where((sdfxyz  == i).all(1))[0][0] for i in vina_pdbqtxyz])

    # Returning the indices to sort sdf atoms to follow original mol2 order
    return ind2[ind1]


def resort_traj(traj, indices): 
    """ Resorts a trajectory from a set of indices. """ 
    table, bonds = traj.topology.to_dataframe()
    # Sort based on indices, then reset the table index 
    table = table.reindex(indices).set_index(np.arange(0,len(table)))
    # replace the old atom index to be ascending i.e 1,2,3....N 
    table["serial"] = np.arange(1,len(table)+1)
    # The new topology from the data frame 
    top = md.Topology.from_dataframe(table,bonds)
    # And then the loading the new traj from the new top! 
    newtraj = md.Trajectory(xyz = traj.xyz[:,indices], topology=top)
    return newtraj



# DOCKING --------------------------------

def run_vina(framedir, traj_frame, lig_pdbqt, boxsize, prep_rec): 
    """ Runs vina and writes optimal binding pose to an out directory for each residue. 
        Returns the out directory with all residue docking files. """

    vinadir = chk_mkdir(f'{framedir}/out', nreturn=True)

    protein_pdb = get_frame_pdb(framedir, traj_frame)
    protein_pdbqt = protein_pdbqt_ADFR(protein_pdb, f'{framedir}/protein.pdbqt', prep_rec)  
    residues = [res.resSeq for res in traj_frame.topology.residues]

    for residue in range(residues[0], residues[-1]): 
        comx, comy, comz = get_res_com(traj_frame, residue) 
        v = Vina(sf_name='vina')
        v.set_receptor(protein_pdbqt) 
        v.set_ligand_from_file(lig_pdbqt)
        v.compute_vina_maps(center=[comx, comy, comz], box_size=[boxsize, boxsize, boxsize]) 
        v.dock(exhaustiveness=32, n_poses=20) 
        v.write_poses(F'{vinadir}/{residue}_out.pdbqt', n_poses=1, overwrite=True) # just write min energy pose
    return vinadir


def dock_all_frames(outdir, traj, pdb, lig_file, prep_rec, ebox): 
    """ Docks for all frames of a trajectory on all residues of the protein. 
        Parameters: outdirectory to write files to
                    path to trajectory file 
                    path to pdb file for the trajectory
                    path to ligand file (mol2, pdb)
                    the prepare_receptor function from the ADFR Suite
                    the eBox script from Feinstein & Brylinski, 2015. """

    dock_name, lig_name = get_filename(traj), get_filename(lig_file)

# Making directories...

    # for docking
    dockdir = chk_mkdir(f'{outdir}/{dock_name}_out', nreturn=True)

    # for out pdb files of the docked ligand
    alloutpdbs = chk_mkdir(f'{outdir}/ligand_outpdbs', nreturn=True)
    outpdbs = chk_mkdir(f'{alloutpdbs}/{dock_name}', nreturn=True)

    # for docking scores 
    score_dir = chk_mkdir(f'{outdir}/scores', nreturn=True)


# Getting files ready for docking 

    lig_pdbqt = mk_lig_pdbqt(lig_file, f'{dockdir}/{lig_name}.pdbqt')
    boxsize = get_boxsize(ebox, lig_pdbqt)
    protein_traj = load_protein_traj(traj, pdb=pdb) 
    print(f"Docking for {protein_traj.n_frames} frames now :) Hang tight!")
    # initializing score array 
    sc_array = np.zeros((protein_traj.n_frames, 2)) # for energy and residue index


# Docking: 

    for frame in range(protein_traj.n_frames): # For all frames
        # Directory for the frame 
        framedir = chk_mkdir(f'{dockdir}/{dock_name}_{frame}', nreturn=True)
        # Running Vina on all residues 
        vina_outdir = run_vina(framedir, protein_traj[frame], lig_pdbqt, boxsize, prep_rec)
        # Getting lowest score for frame and saving info to score array  
        minfile, minenergy, minres = get_lowest_energy(vina_outdir) 
        sc_array[frame, 0] = minenergy
        sc_array[frame, 1] = minres
        # Saving optimal ligand pose as an sdf file, then as pdb (so we can make a trajectory) 
        lig_sdf = meeko_pdbqt_export(minfile, f'{framedir}/ligout.sdf')
        lig_pdb = sdf_to_pdb(lig_sdf, f'{outpdbs}/ligand_{frame}.pdb') 
    # After docking on all frames, saving lowest docking scores for all frames 
    np.save(f'{score_dir}/{dock_name}_scores.npy', sc_array) 


# Making the docked trajectory: 

    # Loading all of the docked ligand pdbs as a trajectory
    ligtraj = md.load(lsdir(outpdbs))
    
    # Reordering the ligand trajectory to match the original mol2 order 
    # Working with what was stored from the for loop, as the indexing shuffles should remain consistent across all frames: 
    indices = get_sdf_indices(lig_file, lig_pdbqt, minfile, lig_sdf) # mol2, mk_pdbqt, vina_pdbqt, sdf 
    newligtraj = resort_traj(ligtraj, indices) 

    # Now stacking to create the new trajectory with the docked ligand... 
    dockedtraj = protein_traj.stack(newligtraj) 

    # Saving to trajoutfiles 
    trajdir = chk_mkdir(f'{outdir}/trajoutfiles', nreturn=True)
    dockedtraj.save_xtc(f'{trajdir}/{dock_name}_out.xtc') 
    dockedtraj[0].save_pdb(f'{trajdir}/{dock_name}_out.pdb') 

    print(f'Docking complete! The docked trajectory file can be found in {trajdir} and the docking data in {dockdir}. Happy docking!') 





if __name__ == '__main__': 


##### ARGPARSE ######
    parser = argparse.ArgumentParser(description = "From a trajectory file, associated pdb and ligand file (mol2), will run AutoDock Vina on each residue of the protein for all protein conformations in the trajectory. Returns optimal binding modes of the ligand for each residue as pdbqt, containing ligand coordinates and free energy of the pose. Will also return a docked trajectory, with each pose corresponding to the most favorable binding pose for each conformation.")
    
    parser.add_argument("--outdir", required=True, type=str, help="The directory to which you'd like to save all the docking data and analysis :)")

    parser.add_argument("--traj", required = True, type = str, help = "The trajectory you wish to dock on. ")
    parser.add_argument("--pdb", type = str, required = True, help = "the corresponding pdb file for the trajectory. ")
    parser.add_argument("--lig_file", type = str, required = True, help = 'The mol2 file of the ligand you would like to dock with.') 

    parser.add_argument("--prep_rec", type = str, required = False, default = './ADFRsuite/bin/prepare_receptor')
    parser.add_argument("--ebox", type=str, required=False, default = './scripts/eBoxSize-1.1.pl') 

    args = parser.parse_args()
################


    dock_all_frames(args.outdir, args.traj, args.pdb, args.lig_file, args.prep_rec, args.ebox)



