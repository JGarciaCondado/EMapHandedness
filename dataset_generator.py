import subprocess
import xmippLib
import os
import string
import random
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from skimage import measure
from tqdm import tqdm

def runJob( cmd, cwd='./'):
    """ Run command in a supbrocess using the xmipp3 environment
        return True if process finished correctly.
    """
    p = subprocess.Popen(cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    p.wait()
    return 0 == p.returncode

def simulate_volume(PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues):
    """ Simulate PDB to create Electron Density Map, obtain a mask by thresholding
        and also include a mask of where the SSE of interest are found
    """
    # Create temporary name
    fnRandom = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(32)])
    fnHash = "nrPDB/tmp"+fnRandom
    # Center pdb
    ok = runJob("xmipp_pdb_center -i %s -o %s_centered.pdb"%(PDB,fnHash))
    # Obtain desired SSE pdb sections
    if ok:
        if SSE_type is 'alpha':
            ok = runJob("xmipp_pdb_select -i %s_centered.pdb -o %s_SSE.pdb --keep_alpha %d"%(fnHash,fnHash,minresidues))
        elif SSE_type is 'beta':
            ok = runJob("xmipp_pdb_select -i %s_centered.pdb -o %s_SSE.pdb --keep_beta %d"%(fnHash,fnHash,minresidues))
        else:
            ok = False
    # Sample whole pdb
    if ok:
        ok = runJob("xmipp_volume_from_pdb  -i %s_centered.pdb -o %s --sampling 1 -v 0"%(fnHash,fnHash))
    # Filter to maxRes
    if ok:
        ok = runJob("xmipp_transform_filter -i %s.vol -o %sFiltered.vol --fourier low_pass %f 0.02 --sampling 1 -v 0"%(fnHash,fnHash,maxRes))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %sFiltered.vol -o %sMask.vol --select below %f --substitute binarize -v 0"%(fnHash,fnHash,mask_threshold))
    # Obtain volumes
    if ok:
        Vf = xmippLib.Image("%sFiltered.vol"%fnHash).getData()
        Vmask = xmippLib.Image("%sMask.vol"%fnHash).getData()
    else:
        Vf, Vmask = None, None

    if Vf is not None and Vmask is not None:
        ok = runJob("xmipp_volume_from_pdb  -i %s_SSE.pdb -o %s_SSE --sampling 1 --size %d -v 0"%(fnHash,fnHash, Vf.shape[0]))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %s_SSE.vol -o %sMask_SSE.vol --select below %f --substitute binarize -v 0"%(fnHash,fnHash,SSE_mask_threshold))
    # Save mask
    if ok:
        Vmask_SSE = xmippLib.Image("%sMask_SSE.vol"%fnHash).getData()
    else:
        Vmask_SSE = None
    #Remove all temporary files produced 
    os.system("rm -f %s*"%fnHash)

    return Vf, Vmask, Vmask_SSE

def get_SSE_centroids(Vmask_SSE):
    """ From the SSE mask obtain the centroids of each of the SSE elements identified.
    """
    # Erode mask so as to retain center of SSE and seperate SSE that might have merged in sampling
    Vmask_SSE_outline = binary_erosion(Vmask_SSE, structure=np.ones((3,3,3))).astype(Vmask_SSE.dtype)
    # Label different regions
    Vmask_SSE_objects = measure.label(Vmask_SSE_outline)
    # Create object that defines region properties
    Vmask_SSE_regions = measure.regionprops(Vmask_SSE_objects, cache=False)
    # Array to store centroid values rounded
    SSE_centroids = []
    # Obtain centroids for each region
    for region in Vmask_SSE_regions:
        # Remove small objects that are probably small disconnections from main SSE
        if region['area'] > 50:
            centroid = [np.rint(i).astype('int') for i in region['centroid']]
            # Check that centroid is also inside outline
            if Vmask_SSE_outline[centroid[0], centroid[1], centroid[2]]:
                SSE_centroids.append(centroid)

    return SSE_centroids

def extract_boxes(Vf, centroids, box_dim):
    """ Given a set of cordinates and dimensions extract boxes at that point
    """
    boxes = []
    # Box half width assumes dimension must be odd
    box_hw = int((box_dim-1)/2)
    for centroid in centroids:
        boxes.append(Vf[centroid[0]-box_hw:centroid[0]+box_hw+1,
                        centroid[1]-box_hw:centroid[1]+box_hw+1,
                        centroid[2]-box_hw:centroid[2]+box_hw+1])
    return boxes

def get_mask_no_SSE(Vmask, Vmask_SSE, SE):
    """ Obtain a mask that contains those parts that do not have the SSE of interest
    """
    # First obtain Not SSE
    Vmask_not_SSE = np.logical_not(Vmask_SSE).astype(Vmask.dtype)
    # Erode not SSE with SE to avoid choosing boxes close to SSE
    Vmask_not_SSE_eroded = binary_erosion(Vmask_not_SSE, structure=SE).astype(Vmask.dtype)
    # Find union with Vmask so that areas away from SSE remain unchanged
    Vmask_no_SSE = np.logical_and(Vmask, Vmask_not_SSE_eroded).astype(Vmask.dtype)

    return Vmask_no_SSE

def get_no_SSE_centroids(Vmask_no_SSE, n_centroids):
    """ Randomly select n centroids from mask
    """
    # Obtain coordinates where mask is 1
    possible_centroids = np.argwhere(Vmask_no_SSE == 1.0)
    # Randomly choose n of this
    if len(possible_centroids)>0:
        centroid_ids = np.random.choice(len(possible_centroids), n_centroids)
        return possible_centroids[centroid_ids]
    else:
        return None

def extract_boxes_PDB(PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues, box_dim):
    """ For a PDB extract SSE helices and boxes not containg SSE helices
    """
    # Get volumes 
    Vf, Vmask, Vmask_SSE = simulate_volume(PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues)
    # Exit if simulation unsuccesfull
    if Vf is None or Vmask is None or Vmask_SSE is None:
        return None, None
    # Get SSE centroids
    SSE_centroids = get_SSE_centroids(Vmask_SSE)
    # Extract SSE boxes
    SSE_boxes = extract_boxes(Vf, SSE_centroids, box_dim)
    # Get volume mask with no SSEs
    Vmask_no_SSE = get_mask_no_SSE(Vmask, Vmask_SSE, SE)
    # Sample centroids from mask containing no SSEs
    no_SSE_centroids = get_no_SSE_centroids(Vmask_no_SSE, len(SSE_centroids))
    # Extract no SSE boxes
    if no_SSE_centroids is not None:
        no_SSE_boxes = extract_boxes(Vf, no_SSE_centroids, box_dim)
    else:
        no_SSE_boxes = None

    return SSE_boxes, no_SSE_boxes

def create_directory(path):
    """ Create directory if it does not exist
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def create_SSE_dataset(data_root, dataset_root, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues, box_dim):
    """ Creat the whole dataset from a directory containg all the PDBS
    """
    # Create dataset direcotry if it doesn't exist
    create_directory(dataset_root)
    for PDB in tqdm(os.listdir(data_root)):
        # Ensure we are working with pdb files
        if PDB[-4:] != '.pdb':
            continue
        # If PDB dataset already there in case errors cause restart
        if os.path.isdir(dataset_root+PDB[:-4]):
            continue
        #Obtain boxes
        SSE_boxes, no_SSE_boxes = extract_boxes_PDB(data_root+PDB, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues, box_dim)
        if SSE_boxes is not None and no_SSE_boxes is not None:
            # Create directory with pdb name
            create_directory(dataset_root+PDB[:-4])
            # Create subdirecotries to store SSE and no SSE
            create_directory(dataset_root+PDB[:-4]+'/'+SSE_type)
            create_directory(dataset_root+PDB[:-4]+'/no_'+SSE_type)
            # Save the different boxes
            for i, box in enumerate(SSE_boxes):
                # Check correct box dimensions it might be boxes were in a corner
                # and have uneven box dimensions
                if box.shape == (box_dim, box_dim, box_dim):
                    np.save(dataset_root+PDB[:-4]+'/%s/'%SSE_type+'box%d.npy'%i, box)
            for i, box in enumerate(no_SSE_boxes):
                if box.shape == (box_dim, box_dim, box_dim):
                    np.save(dataset_root+PDB[:-4]+'/no_%s/'%SSE_type+'box%d.npy'%i, box)

if __name__ == "__main__":
    # Define variables
    data_root = 'nrPDB/PDB/'
    dataset_root = 'nrPDB/Dataset/3A/'
    SSE_type = 'alpha'
    maxRes = 3.0
    mask_threshold = 0.5
    SSE_mask_threshold = 0.5
    minresidues = 7
    box_dim = 11
    SE = np.ones((3,3,3))

    # Create dataset
    create_SSE_dataset(data_root, dataset_root, maxRes, mask_threshold, SSE_mask_threshold, SSE_type, minresidues, box_dim)
