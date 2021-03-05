import subprocess
import xmippLib
import os
import string
import random
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from skimage import measure

def runJob( cmd, cwd='./'):
    """ Run command in a supbrocess using the xmipp3 environment
        return True if process finished correctly.
    """
    p = subprocess.Popen(cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    p.wait()
    return 0 == p.returncode

def simulate_volume(PDB, maxRes, threshold, alpha_threshold, minresidues):
    """ Simulate PDB to create Electron Density Map obtain mask by thresholding
        and also include a mask of where the alpha helix are found
    """
    # Create temporary name
    fnRandom = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(32)])
    fnHash = "nrPDB/tmp"+fnRandom
    # Center pdb
    ok = runJob("xmipp_pdb_center -i %s -o %s_centered.pdb"%(PDB,fnHash))
    # Obtain only alphas
    if ok:
        ok = runJob("xmipp_pdb_select -i %s_centered.pdb -o %s_alpha.pdb --keep_alpha %d"%(fnHash,fnHash,minresidues))
    # Sample whole pdb
    if ok:
        ok = runJob("xmipp_volume_from_pdb  -i %s_centered.pdb -o %s --sampling 1 -v 0"%(fnHash,fnHash))
    # Filter to maxRes
    if ok:
        ok = runJob("xmipp_transform_filter -i %s.vol -o %sFiltered.vol --fourier low_pass %f 0.02 --sampling 1 -v 0"%(fnHash,fnHash,maxRes))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %sFiltered.vol -o %sMask.vol --select below %f --substitute binarize -v 0"%(fnHash,fnHash,threshold))
    # Obtain volumes
    if ok:
        Vf = xmippLib.Image("%sFiltered.vol"%fnHash).getData()
        Vmask = xmippLib.Image("%sMask.vol"%fnHash).getData()
    else:
        Vf, Vmask = None, None

    if Vf is not None and Vmask is not None:
        ok = runJob("xmipp_volume_from_pdb  -i %s_alpha.pdb -o %s_alpha --sampling 1 --size %d -v 0"%(fnHash,fnHash, Vf.shape[0]))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %s_alpha.vol -o %sMask_alpha.vol --select below %f --substitute binarize -v 0"%(fnHash,fnHash,alpha_threshold))
    # Save mask
    if ok:
        Vmask_alpha = xmippLib.Image("%sMask_alpha.vol"%fnHash).getData()
    else:
        Vmask_alpha = None
    #Remove all temporary files produced 
    os.system("rm -f %s*"%fnHash)

    return Vf, Vmask, Vmask_alpha

def get_alpha_centroids(Vmask_alpha):
    """ From the alpha mask obtain the centroids of each othe alpha helices
    """
    # Erode mask so as to retain center of alpha helix and seperate helices that might have merged in sampling
    Vmask_alpha_outline = binary_erosion(Vmask_alpha, structure=np.ones((3,3,3))).astype(Vmask_alpha.dtype)
    # Label different regions
    Vmask_alpha_objects = measure.label(Vmask_alpha_outline)
    # Create object that defines region properties
    Vmask_alpha_regions = measure.regionprops(Vmask_alpha_objects, cache=False)
    # Array to store centroid values rounded
    alpha_centroids = []
    # Obtain centroids for each region
    for region in Vmask_alpha_regions:
        # Remove small objects that are probably small disconnections from main alpha
        if region['area'] > 50:
            centroid = [np.rint(i).astype('int') for i in region['centroid']]
            # Check that centroid is also inside outline
            if Vmask_alpha_outline[centroid[0], centroid[1], centroid[2]]:
                alpha_centroids.append(centroid)

    return alpha_centroids

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

def get_mask_no_alpha(Vmask, Vmask_alpha, SE):
    """ Obtain a mask that contains those parts that do not have alpha helices
    """
    # First obtain Not Alpha
    Vmask_not_alpha = np.logical_not(Vmask_alpha).astype(Vmask.dtype)
    # Erode not alpha with SE
    Vmask_not_alpha_eroded = binary_erosion(Vmask_not_alpha, structure=SE).astype(Vmask.dtype)
    # Find union with Vmask so that areas away from alpha remain unchanged
    Vmask_no_alpha = np.logical_and(Vmask, Vmask_not_alpha_eroded).astype(Vmask.dtype)

    return Vmask_no_alpha

def get_no_alpha_centroids(Vmask_no_alpha, n_centroids):
    """ Randomly select n centroids from mask
    """
    # Obtain coordinates where mask is 1
    possible_centroids = np.argwhere(Vmask_no_alpha == 1.0)
    # Randomly choose n of this
    centroid_ids = np.random.choice(len(possible_centroids), n_centroids)

    return possible_centroids[centroid_ids]

def extract_boxes_PDB(PDB, maxRes, threshold, alpha_threshold, minresidues, box_dim):
    """ For a PDB extract alpha helices and boxes not containg alpha helices
    """
    # Get volumes 
    Vf, Vmask, Vmask_alpha = simulate_volume(PDB, maxRes, threshold, alpha_threshold, minresidues)
    # Get alpha centroids
    alpha_centroids = get_alpha_centroids(Vmask_alpha)
    # Extract alpha boxes
    alpha_boxes = extract_boxes(Vf, alpha_centroids, box_dim)
    # Get volume mask with no alphas
    Vmask_no_alpha = get_mask_no_alpha(Vmask, Vmask_alpha, SE)
    # Sample centroids from mask containing no alphas
    no_alpha_centroids = get_no_alpha_centroids(Vmask_no_alpha, len(alpha_centroids))
    # Extract no alpha boxes
    no_alpha_boxes = extract_boxes(Vf, no_alpha_centroids, box_dim)

    return alpha_boxes, no_alpha_boxes

def create_directory(path):
    """ Create directory if it does not exist
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def create_alpha_dataset(data_root, dataset_root, maxRes, threshold, alpha_threshold, minresidues, box_dim):
    """ Creat the whole dataset from a directory containg all the PDBS
    """
    # Create dataset direcotry if it doesn't exist
    create_directory(dataset_root)
    for PDB in os.listdir(data_root):
        # Ensure we are working with pdb files
        if PDB[-4:] != '.pdb':
            continue
        #Obtain boxes
        alpha_boxes, no_alpha_boxes = extract_boxes_PDB(data_root+PDB, maxRes, threshold, alpha_threshold, minresidues, box_dim)
        # Create directory with pdb name
        create_directory(dataset_root+PDB[:-4])
        # Create subdirecotries to store alpha and no alpha
        create_directory(dataset_root+PDB[:-4]+'/alpha')
        create_directory(dataset_root+PDB[:-4]+'/no_alpha')
        # Save the different boxes
        for i, box in enumerate(alpha_boxes):
            np.save(dataset_root+PDB[:-4]+'/alpha/'+'box%d.npy'%i, box)
        for i, box in enumerate(no_alpha_boxes):
            np.save(dataset_root+PDB[:-4]+'/no_alpha/'+'box%d.npy'%i, box)

if __name__ == "__main__":
    # Define variables
    data_root = 'nrPDB/PDB/'
    dataset_root = 'nrPDB/Dataset/'
    maxRes = 5.0
    threshold = 0.5
    alpha_threshold = 0.5
    minresidues = 7
    box_dim = 11
    SE = np.ones((3,3,3))

    # Create dataset
    create_alpha_dataset(data_root, dataset_root, maxRes, threshold, alpha_threshold, minresidues, box_dim)
