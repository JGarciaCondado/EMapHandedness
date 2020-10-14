from tensorflow.keras.utils import Sequence
import glob
import numpy as np
import os
import string
import xmippLib
import xmipp3
import subprocess

class Generator(Sequence):
    """Generator class that extracts simulated electron density volumes from PDBs
    and generates from these the appropriate batches to feed to the CNN.

    Arguments:
    fnDir: directory contianing PDBs to use
    batch_size: # of boxes in a batch
    batch_per_epoch: # of batches in an epoch
    boxDim: integer especifyin box dimension
    maxRes: maximum resolution used in simulation
    """

    def __init__(self, fnDir, batch_size, batch_per_epoch, boxDim, maxRes):
        self.PDBs = glob.glob(fnDir+"/*pdb")
        self.maxRes = maxRes
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch
        self.boxDim = boxDim
        self.Boxes=[]

    def __len__(self):
        return self.batch_per_epoch

    def runJob(self, cmd, cwd='./'):
        """ Run command in a supbrocess using the xmipp3 environment
            return True if process finished correctly.
        """
        p = subprocess.Popen(cmd, cwd=cwd, env=xmipp3.Plugin.getEnviron(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        p.wait()
        return 0 == p.returncode
            
    def simulate_volume(self, PDB, tmp_name):
        """ Use Xmipp library to simulate volumes from PDB models.
        Returns a volume and a mask.
        """
        # Center pdb
        ok = self.runJob("xmipp_pdb_center -i %s -o %s_centered.pdb"%(PDB,tmp_name))
        # Sample pdb
        if ok:
            ok = self.runJob("xmipp_volume_from_pdb  -i %s_centered.pdb -o %s --sampling 1 --size 200 -v 0"%(tmp_name,tmp_name))
        # Filter to maxRes
        if ok:
            ok = self.runJob("xmipp_transform_filter -i %s.vol -o %sFiltered.vol --fourier low_pass %f 0.02 --sampling 1 -v 0"%(tmp_name,tmp_name,self.maxRes))
        # Create mask by thresholding
        if ok:
            ok = self.runJob("xmipp_transform_threshold -i %s.vol -o %sMask.vol --select below 0.3 --substitute binarize -v 0"%(tmp_name,tmp_name))
        # Binarize mask
        if ok:
            ok = self.runJob("xmipp_transform_morphology -i %sMask.vol --binaryOperation erosion -v 0"%tmp_name)
        # Obtain volumes
        if ok:
            Vf = xmippLib.Image("%sFiltered.vol"%tmp_name).getData()
            Vmask = xmippLib.Image("%sMask.vol"%tmp_name).getData()
        else:
            Vf, Vmask = None, None
        
        #Remove all temporary files produced 
        os.system("rm -f %s*"%tmp_name)

        return Vf, Vmask

    def createBoxes(self, Vf, Vmask):
        """ Extract boxes from volume and mask provided
        """
        # TODO fix handling of boxes to deal with even number of dimensions
        Zdim, Ydim, Xdim = Vf.shape
        boxDim2 = self.boxDim//2
        # Iterate over all coordinates in volume and extract a box around these
        for z in range(boxDim2,Zdim-boxDim2):
            for y in range(boxDim2,Ydim-boxDim2):
                for x in range(boxDim2,Xdim-boxDim2):
                    if Vmask[z,y,x]>0:
                        box=Vf[z-boxDim2:z+boxDim2+1,y-boxDim2:y+boxDim2+1,x-boxDim2:x+boxDim2+1]
                        self.Boxes.append(box/np.linalg.norm(box))

    def populate_boxes(self):
        """ Simulate electron density from PDB and extract boxes from the volume. 
        """

        # Choose random name to assing to saved files
        fnRandom = ''.join([np.random.choice([char for char in string.ascii_letters + string.digits]) for i in range(32)])
        fnHash = "tmp"+fnRandom
        
        # Try generating the volume if this fails remove PDB from list and choose a new
        Vf, Vmask = None, None
        while Vf is None:
            # Choose random PDB from all those available
            n = np.random.randint(0,len(self.PDBs))
            PDB = self.PDBs[n]

            #Obtain volumes and mask
            Vf, Vmask = self.simulate_volume(PDB, fnHash)

            # Remove PDB from list if unsuscessful
            if Vf is None:
                del self.PDBs[n]
            else:
                self.createBoxes(Vf, Vmask)

    def __getitem__(self,idx):
        """ Generate a batch 
        """
        batchX = np.zeros((self.batch_size,self.boxDim,self.boxDim,self.boxDim,1))
        batchY = np.zeros(self.batch_size)
        for n in range(self.batch_size):
            # If there are no more boxes populate
            if self.Boxes == []:
                self.populate_boxes()
            box, target = self.getBox()
            batchX[n,:,:,:,0] = box
            batchY[n] = target
        return batchX, batchY


    def getBox(self):
        """ Return a box with equal likelihood of being flipped
        """
        # Pop a box form those available
        box = self.Boxes.pop() 
        # Target 0 means box not flipped
        target = 0
        # Flip box randomly 50% of times
        if np.random.randint(2):
            box = np.flip(box, np.random.randint(0,3))
            target=1
        return box, target
