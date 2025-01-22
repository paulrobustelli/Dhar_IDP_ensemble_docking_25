#!/usr/bin/env python3

#########################
import numpy as np 
import os, sys 
import argparse
import pandas as pd
import subprocess as sp
from openbabel import openbabel
import mdtraj as md
##########################
# convenience functions: 
from .utils import chk_mkdir, lsdir, get_filename
###############


# PREPARING DIFFDOCK FILES 

def make_pdbs_from_traj(outdir, trajfile, pdb, dock_name): 
    """ Will save a pdb for each frame of a trajectory and then return a list of the pdbs for docking and the protein trajectory. """
    pdbdir = chk_mkdir(f'{outdir}/predock_pdbs', nreturn=True)
    trajpdbs = chk_mkdir(f'{pdbdir}/{dock_name}', nreturn=True)
    traj = md.load(trajfile, top=pdb) 
    protein_traj = traj.atom_slice(traj.top.select("protein"))
    for frame in range(protein_traj.n_frames): 
        protein_traj[frame].save_pdb(f'{trajpdbs}/{dock_name}_{frame}.pdb')
    return lsdir(trajpdbs), protein_traj 

 
def diffdock_csv(cwd, pdb_list, lig_path, dock_name): # n_frames, indices, cluster):
    """ Makes the appropriate csv file to run diffdock fron a list of pdbs and a ligand. """
    str_list = lambda x, n: [x]*n
    csv_dir = chk_mkdir(f'{cwd}/run_csvs', nreturn=True)
    out_framedir_paths = [f'{dock_name}_out/{dock_name}_{x}' for x in range(len(pdb_list))] # strings for all the frame dirs for each pdb given
    df = pd.DataFrame({'complex_name': out_framedir_paths, 'protein_path': pdb_list, 'ligand_description': str_list(lig_path, len(pdb_list)), 'protein_sequence': str_list('', len(pdb_list))})
    df.to_csv(f'{csv_dir}/{dock_name}.csv', index=False, header=True, sep=',')
    return f'{csv_dir}/{dock_name}.csv', [f'{cwd}/{x}' for x in out_framedir_paths]


# POST-PROCESSING

def sdf_to_pdb_hyd(sdf, out_file_path): 
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdb")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, sdf) 
    mol.AddHydrogens()
    mol.AddPolarHydrogens()
    obConversion.WriteFile(mol, out_file_path)
    return out_file_path


def parse_dd_outfiles(outdir, framedirs, dock_name): 
    """ Will generate indices of failed docking frames and a list of successfully docked ligand pdbs. Returns directory containing ligand pdbs.""" 

# Making directories

    # docked ligand pdbs
    alloutpdbs = chk_mkdir(f'{outdir}/ligand_outpdbs', nreturn=True) 
    outpdbs = chk_mkdir(f'{alloutpdbs}/{dock_name}', nreturn=True)
    # confidence scores 
    csc_dir = chk_mkdir(f'{outdir}/scores', nreturn=True)
    # indices of failed frames
    faileddir = chk_mkdir(f'{outdir}/failed_ind', nreturn=True)

    # Initializing arrays...
    failed = []
    score = []
    success = []

    # Iterating over all frame directories 
    for frame, framedir in enumerate(framedirs): 
        isExist = os.path.exists(f'{framedir}/rank1.sdf')
        # only if docking did not fail:
        if isExist: 
            ligpdb = sdf_to_pdb_hyd(f'{framedir}/rank1.sdf', f'{outpdbs}/ligout_{frame}.pdb')
            score.append(get_score(framedir))   
            success.append(frame)
        # if failed: 
        else: 
            failed.append(int(frame))

    np.savetxt(f'{faileddir}/{dock_name}_failed.txt', np.array(failed))
    np.save(f'{csc_dir}/{dock_name}_scores.npy', np.array([score, success]))

    return outpdbs, np.array(success)

def make_traj(outdir, protein_traj, ligpdbs, success_ind, dock_name): 
    """ Makes a trajectory given a list of ligand pdbs, indices of the failed frames, and the original trajectory. """
    # Slicing the original protein traj to match 
    sliced_protein_traj = protein_traj[success_ind]
     
    # loading a ligand trajectory from the pdbs and stacking with the sliced protein trajectory 
    ligtraj = md.load(lsdir(ligpdbs))
    docked_traj = sliced_protein_traj.stack(ligtraj)
 
    # Saving...
    outtrajs = chk_mkdir(f'{outdir}/trajoutfiles', nreturn=True) 
    docked_traj.save_xtc(f'{outtrajs}/{dock_name}_out.xtc')
    docked_traj[0].save_pdb(f'{outtrajs}/{dock_name}_out.pdb')  

def get_score(ddoutdir): 
    """ Will return the confidence score for the rank 1 position for a given diffdock out directory. """ 
    rank1file = lsdir(ddoutdir, keyword="rank1_")
    split1 = rank1file[0].split('e')
    split2 = split1[-1].split('.')
    return float(f'{split2[0]}.{split2[1]}')


# DOCKING

def run_diffdock(outdir, git_repo, trajfile, pdb, ligfile): 
    """ Runs Diffdock for all frames of the given trajectory with the ligand. """
    # Preparing files:
    dock_name = get_filename(trajfile)
    predockpdbs, protein_traj = make_pdbs_from_traj(outdir, trajfile, pdb, dock_name)   
    run_csv, framedirs = diffdock_csv(outdir, predockpdbs, ligfile, dock_name)
    print(f"Docking on {protein_traj.n_frames} now. Hang tight! :) ")

    # Running diffdock: 
    sp.run(['python3', f'{git_repo}/inference.py', '--protein_ligand_csv', run_csv, '--out_dir', outdir, 
            '--samples_per_complex', '15', '--model_dir', f'{git_repo}/workdir/paper_score_model', 
            '--confidence_model_dir', f'{git_repo}/workdir/paper_confidence_model']) 

    # Now, parsing out files
    ligout_pdbdir, success_ind = parse_dd_outfiles(outdir, framedirs, dock_name) 

    # Making & saving the docked trajectory for successful frames
    make_traj(outdir, protein_traj, ligout_pdbdir, success_ind, dock_name) 
    print(f"Docking complete! You can find the docked trajectory in {outdir}/trajoutfiles and all docking data in {outdir}/{dock_name}_out. Happy docking!") 


   


if __name__ == '__main__': 

#################
    parser = argparse.ArgumentParser(description = "Runs DiffDock on all frames of a given trajectory (will dock on only the protein, and slice out all other atom types). Docks with a provided ligand mol2 file. Will save docked trajectories and all dock data to a specified out directory.") 

    parser.add_argument("--outdir", required=True, type=str, help='The directory where all the out directories and files will be written :) ') 

    parser.add_argument('--traj', required=True, type=str, help='The path to the traj file you want to use. Will dock for all frames of the trajectory provided.') 
    parser.add_argument('--pdb', required=True, type=str, help='The pdb topology file to properly load the provided trajectory file.') 
    parser.add_argument('--lig_file', required=True, type=str, help='The path to the ligand mol2 file you wish to dock with.') 

    parser.add_argument('--git_repo', required=False, default= './DiffDock', help='DiffDock repository')

    args = parser.parse_args()
################

    run_diffdock(args.outdir, args.git_repo, args.traj, args.pdb, args.lig_file)
    
