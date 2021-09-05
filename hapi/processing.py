"""Define processing functions for simulated and experimental maps."""

import mrcfile
import numpy as np
import os

from .utils import runJob, createHash
from scipy.ndimage.morphology import binary_erosion
from skimage import measure

def mrc_mask_to_binary(Mask, thr=0.01):
    """Mrc files are not binary so have to be converted back a threshold slighlty
    higher than 0 is chosen due to numerical errors"""
    Mask[Mask<thr] = 0
    Mask[Mask>0] = 1
    return Mask

def get_SSE_centroids(Vmask_SSE, SE):
    """From the mask obtain the centroids of each of the SSE identified."""
    # Erode mask so as to retain center of SSE and seperate SSE that might
    # have merged in sampling
    Vmask_SSE_outline = binary_erosion(
        Vmask_SSE, structure=SE).astype(Vmask_SSE.dtype)
    # Label different regions
    Vmask_SSE_objects = measure.label(Vmask_SSE_outline)
    # Create object that defines region properties
    Vmask_SSE_regions = measure.regionprops(Vmask_SSE_objects, cache=False)
    # Array to store centroid values rounded
    SSE_centroids = []
    # Obtain centroids for each region
    for region in Vmask_SSE_regions:
        # Remove small objects that are probably small disconnections of SSE
        if region['area'] > 50:
            centroid = [np.rint(i).astype('int') for i in region['centroid']]
            # Check that centroid is also inside outline
            if Vmask_SSE_outline[centroid[0], centroid[1], centroid[2]]:
                SSE_centroids.append(centroid)

    return SSE_centroids


def extract_boxes(Vf, centroids, box_dim):
    """Given a set of cordinates extract boxes at those point."""
    boxes = []
    # Box half width assumes dimension must be odd
    box_hw = int((box_dim - 1) / 2)
    for centroid in centroids:
        boxes.append(Vf[centroid[0] - box_hw:centroid[0] + box_hw + 1,
                        centroid[1] - box_hw:centroid[1] + box_hw + 1,
                        centroid[2] - box_hw:centroid[2] + box_hw + 1])
    return boxes


def get_mask_no_SSE(Vmask, Vmask_SSE, SE):
    """Obtain a mask that contains no SSE of interest."""
    # First obtain Not SSE
    Vmask_not_SSE = np.logical_not(Vmask_SSE).astype(Vmask.dtype)
    # Erode not SSE with SE to avoid choosing boxes close to SSE
    Vmask_not_SSE_eroded = binary_erosion(
        Vmask_not_SSE, structure=SE).astype(Vmask.dtype)
    # Find union with Vmask so that areas away from SSE remain unchanged
    Vmask_no_SSE = np.logical_and(
        Vmask, Vmask_not_SSE_eroded).astype(Vmask.dtype)

    return Vmask_no_SSE


def get_no_SSE_centroids(Vmask_no_SSE, n_centroids):
    """Randomly select n centroids from mask."""
    # Obtain coordinates where mask is 1
    possible_centroids = np.argwhere(Vmask_no_SSE == 1.0)
    # Randomly choose n of this
    if len(possible_centroids) > 0:
        centroid_ids = np.random.choice(len(possible_centroids), n_centroids)
        return possible_centroids[centroid_ids]
    else:
        return None

def extract_all_boxes(Vf, Vmask, Vmask_SSE, box_dim, SE_centroids,
                      SE_noSSEMask):
    """For a PDB extract SSE helices and boxes not containg SSE helices."""
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


def process_experimental_map(map_file, filter_res, contour_level):
    """Filter and resample experimental maps to given resolution."""

    # Assume all pixel sizes are equal and take x dimension
    with mrcfile.open(map_file) as mrc:
        pixel_size = mrc.voxel_size['x']

    # Create temporary name
    fnHash = createHash()

    # Resize to pixel size of 1A/pixel
    ok = runJob("xmipp_image_resize -i %s -o %sResized.map --factor %f" %
                (map_file, fnHash, pixel_size))
    # Filter to specified resolution
    if ok:
        ok = runJob("xmipp_transform_filter -i %sResized.map -o %sFiltered.map "\
                    "--fourier low_pass %f --sampling 1"
                    % (fnHash, fnHash, filter_res))
    # Get mask by thresholding to conout level provided
    if ok:
        ok = runJob("xmipp_transform_threshold -i %sResized.map -o %sMask.map "\
                    "--select below %f --substitute binarize -v 0" %
                    (fnHash, fnHash, contour_level))
    # Set filtered volume and mask
    if ok:
        with mrcfile.open(fnHash + 'Filtered.map') as mrc:
            Vf = mrc.data.copy()
        with mrcfile.open(fnHash + 'Mask.map') as mrc:
            Vmask = mrc_mask_to_binary(mrc.data.copy())
    else:
        Vf, Vmask = None, None
    # Remove all temporary files produced
    os.system("rm -f %s*" % fnHash)

    return Vf, Vmask
