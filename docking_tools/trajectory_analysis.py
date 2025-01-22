import os, sys
from __future__ import print_function, division
import mdtraj as md
import numpy as np
import math
from numpy import sum
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product 
import scipy.stats as stats


### 1D CONTACTS AND DUAL RESIDUE CONTACTS 

def to_distance_matrix(distances: np.ndarray,
                       n: int,
                       m: int = None,
                       d0: int = 1):
    distances = distances.squeeze()
    assert distances.ndim < 3, "distances must be either 1 or two dimensional"
    distances = distances.reshape(1, -1) if distances.ndim < 2 else distances

    # info about flattened distance matrix
    N, d = distances.shape

    # intra molecular distances
    if m is None:
        matrix = np.zeros([N] + [n] * 2)
        i, j = np.triu_indices(n, d0)
        matrix[:, i, j] = distances
        return (matrix + matrix.transpose(0, 2, 1)).squeeze()

    else:
        assert d == n * m, \
            "Given dimensions (n,m) do not correspond to the dimension of the flattened distances"

        return distances.reshape(-1, n, m).squeeze()
    
def residue_distances(traj,
                      index_0: np.ndarray,
                      index_1: np.ndarray = None):
    # intra distance case
    if index_1 is None:
        indices = combinations(index_0)
        return md.compute_contacts(traj, indices)[0], indices

    # inter distance case
    else:
        indices = product(index_0, index_1)
        dist = md.compute_contacts(traj, indices)[0]
        return to_distance_matrix(dist, dist.shape[0]), indices
    
def dual_contact_(traj, residue_idx):  
    protein = traj.atom_slice(traj.top.select('protein'))
    protein_ligand_distances = residue_distances(traj, np.arange(protein.n_residues), np.array([residue_idx]))
    protein_ligand_contacts = np.where(protein_ligand_distances < 0.6, 1, 0)
    # essentially the covariance matrix of the distances 
    dual = protein_ligand_contacts.T @ protein_ligand_contacts / len(protein_ligand_contacts)
    return dual



### SPECIFIC INTERMOLECULAR INTERACTIONS

# charge contacts
def charge_contacts_(trj, cutoff=0.5, Ligand_Pos_Charges=[], Ligand_Neg_Charges=[]):
    """ Now, not hard-coded! :) """ 
    top = trj.atom_slice(trj.topology.select('protein')).topology
    residues = top.n_residues
    residue_offset = [res.resSeq for res in top.residues][0]
        
    # Grabbing the charged atoms from charged residues in our protein
    Protein_Pos_Charges=top.select("(resname ARG and name CZ) or (resname LYS and name NZ) or (resname HIE and name NE2) or (resname HID and name ND1)")
    Protein_Neg_Charges=top.select("(resname ASP and name CG) or (resname GLU and name CD) or (name OXT) or (resname NASP and name CG)")

    # Now grabbing the resSeq of these charged residues from the atoms 
    neg_res=[]
    pos_res=[]
    
    for i in Protein_Neg_Charges:
        neg_res.append(top.atom(i).residue.resSeq)
    for i in Protein_Pos_Charges:
        pos_res.append(top.atom(i).residue.resSeq)

    # Making pairs for pos lig, neg protein atoms
    charge_pairs_ligpos=[]                      
    for i in Ligand_Pos_Charges:
        for j in Protein_Neg_Charges:              
            charge_pairs_ligpos.append([i,j])

    # Making pais for neg lig, pos protein atoms
    charge_pairs_ligneg=[]                      
    for i in Ligand_Neg_Charges:
        for j in Protein_Pos_Charges:              
            charge_pairs_ligneg.append([i,j])

    # Initializing array 
    Charge_Contacts=np.zeros((trj.n_frames,residues))

    # If there are any pos lig, neg protein interactions, compute contacts:  
    if len(charge_pairs_ligpos) != 0:
        contact  = md.compute_distances(trj, charge_pairs_ligpos)
        contacts = np.asarray(contact).astype(float)
        neg_res_contact_frames=np.where(contacts < cutoff, 1, 0)
        
        for i in range(0,len(neg_res)):
            Charge_Contacts[:,neg_res[i]-residue_offset]=+neg_res_contact_frames[:,i]

    # If there are any neg lig, pos protein interactions, compute contacts: 
    if len(charge_pairs_ligneg) != 0:
        contact  = md.compute_distances(trj, charge_pairs_ligneg)
        contacts = np.asarray(contact).astype(float)
        pos_res_contact_frames=np.where(contacts < cutoff, 1, 0)

        for i in range(0,len(pos_res)):
            Charge_Contacts[:,pos_res[i]-residue_offset]=+pos_res_contact_frames[:,i]
    
    return Charge_Contacts

# hydrophobic interactions
def hphob_contacts_(trj,ligand_residue_index,cutoff=0.4):
    top = trj.topology
    protein_top = trj.atom_slice(trj.topology.select('protein')).topology
    residues = protein_top.n_residues
    residue_offset = [res.resSeq for res in top.residues][0]
    
    # Grabbing all non-aromatic carbon atom ID's in the ligand and the protein 
    ligand_hphob = top.select(f"resid {ligand_residue_index} and element C")
    protein_hphob = top.select(f"resid 0 to {residues-1} and element C and not name CA")

    # Now, making contact pairs between lig & protein atoms:  
    hphob_pairs = []
    for i in ligand_hphob:
        for j in protein_hphob:
            hphob_pairs.append([i, j])

    # Computing contacts 
    contact = np.asarray(md.compute_distances(trj, hphob_pairs))
    contact_frames = np.where(contact < cutoff, 1, 0)
    
    # Cast hydrophobic contacts as per residue in each frame
    Hphob_res_contacts = np.zeros((trj.n_frames, residues))
    for frame in range(trj.n_frames):
        if np.sum(contact_frames[frame]) > 0: # if we have contacts in this frame, 
            contact_pairs = np.where(contact_frames[frame] == 1) # find the pairs that are in contact
            for j in contact_pairs[0]: # for protein res  
                residue = top.atom(hphob_pairs[j][1]).residue.resSeq # recover the protein residue for the atom 
                Hphob_res_contacts[frame][residue-residue_offset] = 1 # add it to the new array we made 
                
    return Hphob_res_contacts

# Functions for aromatic stacking calculations
def normvector_connect(point1, point2): 
    vec = point1-point2
    vec = vec/np.sqrt(np.dot(vec, vec))
    return vec

def angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1))*np.sqrt(np.dot(v2, v2))))
    
def find_plane_normal(points):
    """ Takes a numpy array of xyz coors for all atoms in a ring. 
    Return the plane normal to the ring."""
    N = points.shape[0] # number of atoms 
    A = np.concatenate((points[:, 0:2], np.ones((N, 1))), axis=1)
    B = points[:, 2]
    out = lstsq(A, B, rcond=-1)
    na_c, nb_c, d_c = out[0]
    # math
    if d_c != 0.0:
        cu = 1./d_c
        bu = -nb_c*cu
        au = -na_c*cu
    else:
        cu = 1.0
        bu = -nb_c
        au = -na_c
    normal = np.asarray([au, bu, cu])
    normal /= math.sqrt(math.dot(normal, normal))
    return normal

def find_plane_normal2_assign_atomid(positions, id1=0, id2=1, id3=2):
    # Alternate approach used to check sign - could the sign check cause descrepency with desres?
    v1 = positions[id1]-positions[id2]
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = positions[id3]-positions[id1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal = np.cross(v1, v2)
    return normal

def get_ring_center_normal_assign_atomid(positions, id1=0, id2=1, id3=2):
    """ Takes a numpy array of the xyz coords for all atoms in a ring for one frame. """
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2_assign_atomid(positions, id1, id2, id3)
    # check direction of normal using dot product convention
    comp = np.dot(normal, normal2)
    if comp < 0:
        normal = -normal
    return center, normal

def get_ring_center_normal_trj_assign_atomid(position_array, id1=0, id2=1, id3=2):
    """ Takes a numpy array of shape (n_frames, n_atoms, xyz) and returns the 
    xyz vals for the center of the ring and the point normal to it for each frame. """
    centers_normals = np.zeros((position_array.shape[0], 2, 3)) 
    for i, pos in enumerate(position_array):
        center, normal = get_ring_center_normal_assign_atomid(pos, id1, id2, id3)
        centers_normals[i][0] = center
        centers_normals[i][1] = normal
    return centers_normals

# aromatic stacking
def aro_contacts_(trj, ligand_rings=[], stack_distance_cutoff = 0.65,
                  p_stack_distance_cutoff = 0.65, t_stack_distance_cutoff = 0.75):
    """ Where ligand rings is a list of lists, where each item is a list 
    of atom indices for that particular aromatic ring."""
    
    top = trj.topology
    protein_top = trj.atom_slice(trj.topology.select('protein')).topology
    residues = protein_top.n_residues
    residue_offset = [res.resSeq for res in top.residues][0]

    # the number of aromatic rings in the ligand: 
    ligrings = len(ligand_rings)
    print("Ligand Aromatics Rings:", ligrings)

    # for each aromatic ring, get the ring center and the point normal to it for each frame
    ligand_ring_params = []
    for ring in ligand_rings: 
        positions = trj.xyz[:, np.array(ring), :]
        ligand_centers_normals = get_ring_center_normal_trj_assign_atomid(positions)
        ligand_ring_params.append(ligand_centers_normals)


    # Finding Protein Aromatic Rings
    # Add Apropriate HIS name if there is a charged HIE OR HIP in the structure 
    prot_rings = [] # the atom indices for separate rings in the protein
    prot_ring_name = [] # the residue objects 
    prot_ring_index = [] # the resSeq (121, 122...)  

    # Grabbing atom ids for the aromatic carbons of the protein 
    aro_select = top.select("resname TYR PHE HIS TRP and name CA")
    # Now, grabbing the whole ring's atomids:  
    for i in aro_select:
        atom = top.atom(i)
        resname = atom.residue.name
        if resname == "TYR":
            ring = top.select(
                f'resid {atom.residue.index} and name CG CD1 CD2 CE1 CE2 CZ')
        if resname == "TRP":
            ring = top.select(
                f"resid {atom.residue.index} and name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2")
        if resname == "HIS":
            ring = top.select(f"resid {atom.residue.index} and name CG ND1 CE1 NE2 CD2")
        if resname == "PHE":
            ring = top.select(f"resid {atom.residue.index} and name CG CD1 CD2 CE1 CE2 CZ")

        prot_rings.append(ring) 
        prot_ring_name.append(atom.residue)
        prot_ring_index.append(atom.residue.index+residue_offset)

    # for each aromatic ring in the protein, get the ring center and 
    # the point normal to it for each frame
    prot_ring_params = []
    for atom in prot_rings: 
        positions = trj.xyz[:, np.array(atom), :]
        ring_centers_normals = get_ring_center_normal_trj_assign_atomid(positions)
        prot_ring_params.append(ring_centers_normals)
    
    # The # of rings in the protein
    sidechains = len(prot_rings)
    print(trj.n_frames, sidechains)

    # initializing dictionaries...
    Ringstacked = {}
    Stackparams = {}
    Aro_Contacts = {}
    Pstack = {}
    Tstack = {}

    """
    print("q1: alpha<=45 and beta>=135")
    print("q2: alpha>=135 and beta>=135")
    print("q3: alpha<=45 and beta<=45")
    print("q4: alpha>=135 and beta<=135")
    """

    # considering it one ring at a time 
    for l, ligand_one_ring_param in enumerate(ligand_ring_params): 
        name = f"Lig_ring.{l}"

        Stackparams[name] = {}
        Pstack[name] = {}
        Tstack[name] = {}
        Aro_Contacts[name] = {}

        dists = np.zeros(shape=(trj.n_frames, sidechains)) 
        alphas = np.zeros(shape=(trj.n_frames, sidechains)) 
        betas = np.zeros(shape=(trj.n_frames, sidechains)) 
        thetas = np.zeros(shape=(trj.n_frames, sidechains)) 
        phis = np.zeros(shape=(trj.n_frames, sidechains)) 
        pstacked = np.zeros(shape=(trj.n_frames, sidechains))
        tstacked = np.zeros(shape=(trj.n_frames, sidechains))
        stacked = np.zeros(shape=(trj.n_frames, sidechains))
        aro_contacts = np.zeros(shape=(trj.n_frames, sidechains))
        quadrant=np.zeros(shape=(trj.n_frames,sidechains))

        # for every frame of that one ligand ring
        for i, frame_param in enumerate(ligand_one_ring_param): 
            ligcenter = frame_param[0]
            lignormal = frame_param[1]

            # for each protein ring that we've found 
            for j in range(0, sidechains):
                protcenter = prot_ring_params[j][i,0]
                protnormal = prot_ring_params[j][i,1]

                dists[i, j] = np.linalg.norm(ligcenter-protcenter) 
                connect = normvector_connect(protcenter, ligcenter)
                # alpha is the same as phi in gervasio/Procacci definition
                alphas[i, j] = np.rad2deg(angle(connect, protnormal))
                betas[i, j] = np.rad2deg(angle(connect, lignormal))
                theta = np.rad2deg(angle(protnormal, lignormal))
                thetas[i, j] = np.abs(theta)-2*(np.abs(theta)
                                                > 90.0)*(np.abs(theta)-90.0)
                phi = np.rad2deg(angle(protnormal, connect))
                phis[i, j] = np.abs(phi)-2*(np.abs(phi) > 90.0)*(np.abs(phi)-90.0)

        # iterating through the protein rings again 
        for j in range(0, sidechains):
            name2 = prot_ring_index[j] # the resSeq
            print(f'====> {prot_ring_name[j]}')

            # see where the rings make contact according to our cutoff
            Ringstack = np.column_stack((dists[:, j], alphas[:, j], betas[:, j], thetas[:, j], phis[:, j]))
            r = np.where(dists[:, j] <= stack_distance_cutoff)[0]
            aro_contacts[:, j][r] = 1
            
            r_pstrict = np.where(dists[:, j] <= p_stack_distance_cutoff)[0]
            r_tstrict = np.where(dists[:, j] <= t_stack_distance_cutoff)[0]

            a=np.where(alphas[:,j] >= 135)
            b=np.where(alphas[:,j] <= 45)
            c=np.where(betas[:,j] >= 135)
            d=np.where(betas[:,j] <= 45)
            e=np.where(dists[:,j] <= 0.5)
            q1=np.intersect1d(np.intersect1d(b,c),e)
            q2=np.intersect1d(np.intersect1d(a,c),e)
            q3=np.intersect1d(np.intersect1d(b,d),e)
            q4=np.intersect1d(np.intersect1d(a,d),e)
            stacked[:,j][q1]=1
            stacked[:,j][q2]=1
            stacked[:,j][q3]=1
            stacked[:,j][q4]=1
            quadrant[:,j][q1]=1
            quadrant[:,j][q2]=2
            quadrant[:,j][q3]=3
            quadrant[:,j][q4]=4
            
            # print("q1:",len(q1),"q2:",len(q2),"q3:",len(q3),"q4:",len(q4))
            # print("q1:",len(q1)/total_stacked,"q2:",len(q2)/total_stacked,"q3:",len(q3)/total_stacked,"q4:",len(q4)/total_stacked)
            
            Stackparams[name][name2]=Ringstack
            # print(np.average(Ringstack,axis=0))
            f = np.where(thetas[:, j] <= 45)
            g = np.where(phis[:, j] <= 60)
            h = np.where(thetas[:, j] >= 75)

            pnew = np.intersect1d(np.intersect1d(f, g), r_pstrict)
            tnew = np.intersect1d(np.intersect1d(h, g), r_tstrict)
            pstacked[:, j][pnew] = 1
            tstacked[:, j][tnew] = 1
            stacked[:, j][pnew] = 1
            stacked[:, j][tnew] = 1
            # print("===>Contacts:", len(r), "Total:", total_stacked,"P-stack:", len(pnew), "T-stack:", len(tnew))
            Stackparams[name][name2] = Ringstack

        # Pstack[name] = pstacked
        # Tstack[name] = tstacked
        # Aro_Contacts[name] = aro_contacts
        # Ringstacked[name] = stacked
        # Quadrants[name]=quadrant
    
    aro_res_index = np.array(prot_ring_index)-residue_offset
    aromatic_stacking_contacts_ = np.zeros((trj.n_frames, residues))

    # for each protein aromatic residue, 
    for i in range(0, len(aro_res_index)):
        # for each ligand aromatic ring, 
        for j in range(0, ligrings): 
            # add the ligand interactions from Ringstacked
            aromatic_stacking_contacts_[:, aro_res_index[i]] += Ringstacked[f'Lig_ring.{j}'][:, i]
        
    return aromatic_stacking_contacts_ # , Stackparams

# For hydrogen bond calculation
def _get_bond_triplets(topology, lig_donors, exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
                 if set((one.element.symbol, two.element.symbol)) == elems]
        # Filter non-participating atoms
        atoms = [atom for atom in atoms
                 if can_participate(atom[0]) and can_participate(atom[1])]
        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break  # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')

    nh_donors = get_donors('N', 'H')
    oh_donors = get_donors('O', 'H')
    sh_donors = get_donors('S', 'H')
    xh_donors = np.array(nh_donors + oh_donors + sh_donors+lig_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N', 'S'))
    acceptors = [a.index for a in topology.atoms
                 if a.element.symbol in acceptor_elements and can_participate(a)]
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]

def _compute_bounded_geometry(traj, triplets, distance_cutoff, distance_indices,
                              angle_indices, freq=0.0, periodic=True):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = md.compute_distances(
        traj, triplets[:, distance_indices], periodic=periodic)

    # Now we discover which triplets meet the distance cutoff often enough
    prevalence = np.mean(distances < distance_cutoff, axis=0)
    mask = prevalence > freq

    # Update data structures to ignore anything that isn't possible anymore
    triplets = triplets.compress(mask, axis=0)
    distances = distances.compress(mask, axis=1)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(md.compute_distances(traj, triplets[:, abc_pair],
                                                      periodic=periodic))

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines)  # avoid NaN error
    angles = np.arccos(cosines)
    return mask, distances, angles

def baker_hubbard2(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False,
                   distance_cutoff=0.35, angle_cutoff=150, lig_donor_index=[]):

    angle_cutoff = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets

    # ADD IN LIGAND HBOND DONORS
    add_donors = lig_donor_index

    bond_triplets = _get_bond_triplets(traj.topology,
                                       exclude_water=exclude_water, lig_donors=add_donors, sidechain_only=sidechain_only)

    mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
                                                        distance_cutoff, [1, 2], [0, 1, 2], freq=freq, periodic=periodic)
    # Find triplets that meet the criteria
    presence = np.logical_and(
        distances < distance_cutoff, angles > angle_cutoff)
    mask[mask] = np.mean(presence, axis=0) > freq
    return bond_triplets.compress(mask, axis=0)

# hydrogen bonds
def hbond_(trj, ligand_residue_index, lig_hbond_donors=[]):
    top = trj.topology
    protein_top = trj.atom_slice(trj.topology.select('protein')).topology
    residues = protein_top.n_residues
    residue_offset = [res.resSeq for res in top.residues][0]
    
    # Select Ligand atoms
    ligand = top.select(f"resid {ligand_residue_index}")
    # Select Protein atoms
    protein = top.select(f"resid 0 to {residues-1}")

    # Initializing np.arrays // dicts
    HBond_PD = np.zeros((trj.n_frames, residues))
    HBond_LD = np.zeros((trj.n_frames, residues))
    Hbond_pairs_PD = {}
    Hbond_pairs_LD = {}

    def add_hbond_pair(donor, acceptor, hbond_pairs, donor_res):
        if donor_res not in hbond_pairs:
            hbond_pairs[donor_res] = {}
        if donor not in hbond_pairs[donor_res]:
            hbond_pairs[donor_res][donor] = {}
        if acceptor not in hbond_pairs[donor_res][donor]:
            hbond_pairs[donor_res][donor][acceptor] = 0
        hbond_pairs[donor_res][donor][acceptor] += 1

    # For all frames in the trajectory, 
    for frame in range(trj.n_frames):
        hbonds = baker_hubbard2(trj[frame], angle_cutoff=150, distance_cutoff=0.35, 
                                lig_donor_index=lig_hbond_donors)

        for hbond in hbonds:
            if ((hbond[0] in protein) and (hbond[2] in ligand)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc = top.atom(hbond[2])
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_PD[frame][donor_res-residue_offset] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_PD, donor_res)
            if ((hbond[0] in ligand) and (hbond[2] in protein)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc_id = hbond[2]
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_LD[frame][acc_res-residue_offset] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_LD, acc_res)

    return HBond_PD+HBond_LD


### COMPUTING ANGLES
def angle(x1, x2, x3): 
    v1 = x1 - x2
    v2 = x3 - x2
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0))


def ci(x, iv:float=0.95):
    
    l,r = stats.t.interval(confidence=iv,df = len(x)-1, loc = 0, scale=stats.sem(x))
    return (abs(l)+abs(r))/2

def boot_ci(x,iv=.95,n_resamples=10000,fxn=np.mean, axis=0):
    mean = fxn(x)
    x = (x,)
    l,r = stats.bootstrap(data=x,statistic=fxn,confidence_level = iv,n_resamples=n_resamples,
                           method="percentile",vectorized=True,axis=axis).confidence_interval
    l,r = [i-mean for i in [l,r]]
    return (abs(l)+abs(r))/2


