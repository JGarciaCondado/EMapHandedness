==========================
HaPi - Handedness Pipeline
==========================

HaPi (Handedness Pipeline) is a pipeline used to determine the hand of cryoEM determined maps. Resolving 3D sturctures of macromolecules using Single Particle Analysis (SPA) is an ill-posed problem and so the determined structure can be the specular (mirrored) version of the true underlying structure. Macromolecules in nature have a specific handedness and it is important for atomic fitting of the structure that the resolved structure has the correct hand. At high resolution this is easily determined by looking at the map in a viewer like Chimera but it can be really difficult at low resolutions of 4 to 5 Angstroms. HaPi is a deep learning pipeline to automatically determine the hand of the electron density map automatically for structures below 5 Angstroms of resolution.

=====
Setup
=====

- **Setup enviroment with conda**

Using the command line:

.. code-block:: shell

    conda create -n hapi_env python=3.8.5
    conda activate hapi_env
    python -m pip install -r requirements.txt

- **Install module locally**

Using the command line:

.. code-block:: shell

    python setup.py install

- **Install Xmipp**

The simulator and the experimental map processing require that Xmipp_ commands be available in the command line.

.. _Xmipp: http://xmipp.i2pc.es/

1. Download Scipion_.

.. _Scipion: https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html

2. Make Xmipp commands available.

Using the command line:

.. code-block:: shell

    source path/for/scipion/xmipp-bundle/build/xmipp.bashrc

=========
Structure
=========

The are four main directories in this project:

- **/data**: contains the required scripts and list of ids to download the data used for training and testing.
- **/hapi**: package that contains a set of objects and functions to simulate data, train and test models.
- **/scripts**: set of simple self-descriptive scripts using the hapi package to simulate data, train and test models.
- **/models**: final models trained to be used by users.

====
Data
====

To download the data used a set of batch-download shell scripts and the PDB and EMD IDs are available in **/data**.

- Download PDBs used to simulate boxes

Using the command line:

.. code-block:: shell

   chmod +x batch_download_pdb.sh 
   ./batch_download_pdb.sh -f PDB_boxes_list.txt -o path/for/pdbs -p
   gunzip path/for/pdbs/*.pdb.gz

- Download PDBs used to simulate volumes

Using the command line:

.. code-block:: shell

   chmod +x batch_download_pdb.sh 
   ./batch_download_pdb.sh -f PDB_volumes_list.txt -o path/for/pdbs -p
   gunzip path/for/pdbs/*.pdb.gz

- Download Experimental cryoEM maps

Using the command line:

.. code-block:: shell

   chmod +x batch_download_emdb.sh 
   ./batch_download_emdb.sh -f emdb_list.txt -o path/for/emdmaps

