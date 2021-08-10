import mrcfile
import os
import random
import string

from dataset_generator import runJob

def process_experimental_map(map_file, filter_res):

    # Assume all pixel sizes are equal and take x dimension
    with mrcfile.open(map_file) as mrc:
        pixel_size = mrc.voxel_size['x']
    # Create temporary name
    fnRandom = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(32)])
    fnHash = "tmp"+fnRandom

    # Resize to pixel size of 1A/pixel
    ok = runJob("xmipp_image_resize -i %s -o %sResized.map --factor %f" % (map_file, fnHash, pixel_size))
    # Filter to specified resolution
    if ok:
        ok = runJob("xmipp_transform_filter -i %sResized.map -o %sFiltered.map --fourier low_pass %f --sampling 1 "%(fnHash, fnHash, filter_res))
    # Otsu segementation to obtain mask from non-filtered mask
    if ok:
        ok = runJob("xmipp_volume_segment -i %sResized.map -o %sMask.map --method otsu"%(fnHash, fnHash))
    # Set filtered volume and mask 
    if ok:
        with mrcfile.open(fnHash+'Filtered.map') as mrc:
            Vf = mrc.data.copy()
        with mrcfile.open(fnHash+'Mask.map') as mrc:
            Vmask = mrc.data.copy()
    else:
        Vf, Vmask = None, None
    #Remove all temporary files produced 
    os.system("rm -f %s*"%fnHash)

    return Vf, Vmask

if __name__ == "__main__":
    map_file = 'nrPDB/Exp_maps/maps/emd_11610.map'
    filter_res = 5.0
    Vf, Vmask = process_experimental_map(map_file, filter_res)
