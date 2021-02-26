import subprocess
import xmippLib
import os
import sys
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from skimage import measure
np.set_printoptions(threshold=sys.maxsize)

def runJob( cmd, cwd='./'):
    """ Run command in a supbrocess using the xmipp3 environment
        return True if process finished correctly.
    """
    p = subprocess.Popen(cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    p.wait()
    return 0 == p.returncode

def simulate_volume(PDB, tmp_name, maxRes, threshold, alpha_threshold, minresidues):
    # Center pdb
    ok = runJob("xmipp_pdb_center -i %s -o %s_centered.pdb"%(PDB,tmp_name))
    # Obtain only alphas
    if ok:
        ok = runJob("xmipp_pdb_select -i %s_centered.pdb -o %s_alpha.pdb --keep_alpha %d"%(tmp_name,tmp_name,minresidues))
    # Sample whole pdb
    if ok:
        ok = runJob("xmipp_volume_from_pdb  -i %s_centered.pdb -o %s --sampling 1 -v 0"%(tmp_name,tmp_name))
    # Filter to maxRes
    if ok:
        ok = runJob("xmipp_transform_filter -i %s.vol -o %sFiltered.vol --fourier low_pass %f 0.02 --sampling 1 -v 0"%(tmp_name,tmp_name,maxRes))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %sFiltered.vol -o %sMask.vol --select below %f --substitute binarize -v 0"%(tmp_name,tmp_name,threshold))
    # Obtain volumes
    if ok:
        Vf = xmippLib.Image("%sFiltered.vol"%tmp_name).getData()
        Vmask = xmippLib.Image("%sMask.vol"%tmp_name).getData()
    else:
        Vf, Vmask = None, None

    if Vf is not None and Vmask is not None:
        ok = runJob("xmipp_volume_from_pdb  -i %s_alpha.pdb -o %s_alpha --sampling 1 --size %d -v 0"%(tmp_name,tmp_name, Vf.shape[0]))
    # Create mask by thresholding
    if ok:
        ok = runJob("xmipp_transform_threshold -i %s_alpha.vol -o %sMask_alpha.vol --select below %f --substitute binarize -v 0"%(tmp_name,tmp_name,alpha_threshold))
    # Save mask
    if ok:
        Vmask_alpha = xmippLib.Image("%sMask_alpha.vol"%tmp_name).getData()
    else:
        Vmask_alpha = None
    #Remove all temporary files produced 
    os.system("rm -f %s*"%tmp_name)

    return Vf, Vmask, Vmask_alpha

def get_alpha_centroids(Vmask_alpha):
    # Erode mask so as to retain center of alpha helix and seperate different helices
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
    boxes = []
    # Box half width assumes dimension must be odd
    box_hw = int((box_dim-1)/2)
    for centroid in centroids:
        boxes.append(Vf[centroid[0]-box_hw:centroid[0]+box_hw,
                        centroid[1]-box_hw:centroid[1]+box_hw,
                        centroid[2]-box_hw:centroid[2]+box_hw])
    return boxes

def get_mask_no_alpha(Vmask, Vmask_alpha, SE):
    # Union of not alpha and mask gives those that have no alpha
    Vmask_no_alpha = np.logical_and(Vmask, np.logical_not(Vmask_alpha).astype(Vmask.dtype)).astype(Vmask.dtype)
    # Erode to minimize overlaps with with alpha boxes
    Vmask_no_alpha = binary_erosion(Vmask_no_alpha, structure=np.ones((3,3,3))).astype(Vmask.dtype)

    return Vmask_no_alpha

def get_no_alpha_centroids(Vmask_no_alpha, n_centroids):
    # Obtain coordinates where mask is 1
    possible_centroids = np.argwhere(Vmask_no_alpha == 1.0)
    # Randomly choose n of this
    centroid_ids = np.random.choice(len(possible_centroids), n_centroids)

    return possible_centroids[centroid_ids]

if __name__ == "__main__":
    # Define variables
    PDB = 'nrPDB/PDB/1AGC.pdb'
    tmp_name = 'nrPDB/Examples/1AGC'
    maxRes = 5.0
    threshold = 0.5
    alpha_threshold = 0.5
    minresidues = 7
    box_dim = 11
    SE = np.ones((7,7,7))

    # Get volumes 
    Vf, Vmask, Vmask_alpha = simulate_volume(PDB, tmp_name, maxRes, threshold, alpha_threshold, minresidues)

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
