"""Define functions to simulate data from PDBs."""

import os
import numpy as np
import mrcfile

from tqdm import tqdm
from .utils import *
from .processing import *


def simulate_volume(PDB, maxRes, mask_threshold, SSE_mask_threshold,
                    SSE_type, minresidues):
    """Simulate a cyroEM from a given PDB.

    Simulate PDB to create Electron Density Map, obtain a mask by thresholding
    and also include a mask of where the SSE of interest are found.

    Parameters:
    -----------
    PDB -- Path to pdb file.
    maxRes -- Resolution to filter volume at.
    mask_threshold -- Threshold to obtain non-background voxels.
    SSE_mask_threshold -- Threshold to obtain non-background voxels of SSE
    SSE_type -- Either alpha or beta
    minresidues -- Minimum number of residues to identify it as an SSE.
    """
    # Create temporary name
    fnHash = createHash()
    # Center pdb
    ok = runJob("xmipp_pdb_center -i %s -o %s_centered.pdb" % (PDB, fnHash))
    # Obtain desired SSE pdb sections
    if ok:
        if SSE_type == 'alpha':
            ok = runJob("xmipp_pdb_select -i %s_centered.pdb -o %s_SSE.pdb 
                        --keep_alpha %d" % (fnHash, fnHash, minresidues))
        elif SSE_type == 'beta':
            ok = runJob("xmipp_pdb_select -i %s_centered.pdb -o %s_SSE.pdb 
                        --keep_beta %d" % (fnHash, fnHash, minresidues))
        else:
            ok = False
    # Sample whole pdb
    if ok:
        ok = runJob("xmipp_volume_from_pdb -i %s_centered.pdb -o %s 
                    --sampling 1 -v 0" % (fnHash, fnHash))
    # Filter to maxRes
    if ok:
        ok = runJob("xmipp_transform_filter -i %s.vol -o %sFiltered.map
                    --fourier low_pass %f 0.02 --sampling 1 -v 0" %
                    (fnHash, fnHash, maxRes))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %sFiltered.map -o %sMask.map
                    --select below %f --substitute binarize -v 0" %
                    (fnHash, fnHash, mask_threshold))
    # Obtain volumes
    if ok:
        with mrcfile.open("%sFiltered.map" % fnHash) as mrc:
            Vf = mrc.data.copy()
        with mrcfile.open("%sMask.map" % fnHash) as mrc:
            Vmask = mrc.data.copy()
    else:
        Vf, Vmask = None, None

    if Vf is not None and Vmask is not None:
        ok = runJob("xmipp_volume_from_pdb  -i %s_SSE.pdb -o %s_SSE --sampling
                    1 --size %d -v 0" % (fnHash, fnHash, Vf.shape[0]))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %s_SSE.vol -o %sMask_SSE.map
                    --select below %f --substitute binarize -v 0" %
                    (fnHash, fnHash, SSE_mask_threshold))
    # Save mask
    if ok:
        with mrcfile.open("%sMask_SSE.map" % fnHash) as mrc:
            Vmask_SSE = mrc.data.copy()
    else:
        Vmask_SSE = None
    # Remove all temporary files produced
    os.system("rm -f %s*" % fnHash)

    return Vf, Vmask, Vmask_SSE


def extract_boxes_PDB(PDB, maxRes, mask_threshold, SSE_mask_threshold,
                      SSE_type, minresidues, box_dim, SE_centroids,
                      SE_noSSEMask):
    """For a PDB extract SSE helices and boxes not containg SSE helices."""
    # Get volumes
    Vf, Vmask, Vmask_SSE = simulate_volume(
        PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
    # Exit if simulation unsuccesfull
    if Vf is None or Vmask is None or Vmask_SSE is None:
        return None, None
    # Get SSE centroids
    SSE_centroids = get_SSE_centroids(Vmask_SSE, SE_centroids)
    # Extract SSE boxes
    SSE_boxes = extract_boxes(Vf, SSE_centroids, box_dim)
    # Get volume mask with no SSEs
    Vmask_no_SSE = get_mask_no_SSE(Vmask, Vmask_SSE, SE_noSSEMask)
    # Sample centroids from mask containing no SSEs
    no_SSE_centroids = get_no_SSE_centroids(Vmask_no_SSE, len(SSE_centroids))
    # Extract no SSE boxes
    if no_SSE_centroids is not None:
        no_SSE_boxes = extract_boxes(Vf, no_SSE_centroids, box_dim)
    else:
        no_SSE_boxes = None

    return SSE_boxes, no_SSE_boxes


def create_SSE_dataset(data_root, dataset_root, maxRes, mask_threshold,
                       SSE_mask_threshold, SSE_type, minresidues, box_dim,
                       SE_centroids, SE_noSSEMask, restart=False):
    """Create the dataset of boxesfrom a directory containg all the PDBs

    Parameters:
    -----------
    data_root -- Directory containg all .pdb files.
    dataset_root -- Directory to save dataset to.
    maxRes -- Resolution to filter volume at.
    mask_threshold -- Threshold to obtain non-background voxels.
    SSE_mask_threshold -- Threshold to obtain non-background voxels of SSE
    SSE_type -- Either alpha or beta
    minresidues -- Minimum number of residues to identify it as an SSE.
        Recommended: alpha use 7 as this are two turns
                     beta use 4 as the sheets are smaller
    boxdim -- Odd integer that defines box size (recommended 11)
    SE_centroids -- Sturcture element to obtain better centroids
        Recommended: alpha np.ones((3,3,3)) as they are bulkier
                     beta np.ones((2,2,2)) as they are thinner structures.
    SE_noSSEMask -- Structure element to avoid capturing small part of SSE
    restart -- Restart whole dataset simulation (default False)
    """
    # Create dataset direcotry if it doesn't exist
    create_directory(dataset_root)
    for PDB in tqdm(os.listdir(data_root)):
        # Ensure we are working with pdb files
        if PDB[-4:] != '.pdb':
            continue
        # If PDB dataset already there in case errors cause restart
        # Ignore if we want to redo the complete dataset
        if os.path.isdir(dataset_root+PDB[:-4]) and not restart:
            continue
        # Obtain boxes
        SSE_boxes, no_SSE_boxes = extract_boxes_PDB(data_root+PDB, maxRes,
            mask_threshold, SSE_mask_threshold, SSE_type, minresidues,
            box_dim, SE_centroids, SE_noSSEMask)
        if SSE_boxes is not None and no_SSE_boxes is not None:
            # Create directory with pdb name
            create_directory(dataset_root+PDB[:-4])
            # Create subdirecotries to store SSE and no SSE
            create_directory(dataset_root+PDB[:-4]+'/'+SSE_type)
            create_directory(dataset_root+PDB[:-4]+'/no_'+SSE_type)
            # Save the different boxes
            for i, box in enumerate(SSE_boxes):
                # Check correct box dimensions it might be boxes were in a
                # corner and have uneven box dimensions
                if box.shape == (box_dim, box_dim, box_dim):
                    np.save(dataset_root+PDB[:-4]+'/%s/' %
                            SSE_type+'box%d.npy' % i, box)
            for i, box in enumerate(no_SSE_boxes):
                if box.shape == (box_dim, box_dim, box_dim):
                    np.save(dataset_root+PDB[:-4]+'/no_%s/' %
                            SSE_type+'box%d.npy' % i, box)


def create_volume_dataset(data_root, dataset_root, maxRes, mask_threshold,
                          SSE_mask_threshold, SSE_type, minresidues,
                          restart=False):
    """Create dataset of volumes from a directory containg all the PDBs

    Parameters:
    -----------
    data_root -- Directory containg all .pdb files.
    dataset_root -- Directory to save dataset to.
    maxRes -- Resolution to filter volume at.
    mask_threshold -- Threshold to obtain non-background voxels.
    SSE_mask_threshold -- Threshold to obtain non-background voxels of SSE
    SSE_type -- Either alpha or beta
    minresidues -- Minimum number of residues to identify it as an SSE.
        Recommended: alpha use 7 as this are two turns
                     beta use 4 as the sheets are smaller
    restart -- Restart whole dataset simulation (default False)
    """
    # Create dataset direcotry if it doesn't exist
    create_directory(dataset_root)
    for PDB in tqdm(os.listdir(data_root)):
        # Ensure we are working with pdb files
        if PDB[-4:] != '.pdb':
            continue
        # If PDB dataset already there in case errors cause restart
        # Ignore if we want to redo the complete dataset
        if os.path.isfile(dataset_root+'/%s.npy' % PDB[:-4]) and not restart:
            continue
        # Obtain volumes
        Vf, Vmask, Vmask_SSE = simulate_volume(data_root+PDB, maxRes,
            mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
        if Vf is not None and Vmask is not None and Vmask_SSE is not None:
            # Save all volumes
            data = np.stack([Vf, Vmask, Vmask_SSE])
            np.save(dataset_root+'/%s.npy' % PDB[:-4], data)
