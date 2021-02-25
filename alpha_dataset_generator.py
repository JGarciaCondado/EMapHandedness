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
#    os.system("rm -f %s*"%tmp_name)

    return Vf, Vmask, Vmask_alpha

def get_alpha_centroids(Vmask_alpha):
    # Erode mask so as to retain center of alpha helix and seperate different helices
    Vmask_alpha_outline = binary_erosion(Vmask_alpha, structure=np.ones(box_dim)).astype(Vmask_alpha.dtype)
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
            alpha_centroids.append(centroid)

    return alpha_centroids

if __name__ == "__main__":
    # Define variables
    PDB = 'nrPDB/1AGC.pdb'
    tmp_name = 'nrPDB/test'
    maxRes = 5.0
    threshold = 0.5
    alpha_threshold = 0.5
    minresidues = 7
    box_dim = 11

    # Get volumes 
    Vf, Vmask, Vmask_alpha = simulate_volume(PDB, tmp_name, maxRes, threshold, alpha_threshold, minresidues)

    # Get alpha centroids
    alpha_centroids = get_alpha_centroids(Vmask_alpha)
