==========================
HaPi - Handedness Pipeline
==========================

HaPi (Handedness Pipeline) is a pipeline used to determine the hand of cryoEM determined maps. Resolving 3D sturctures of macromolecules using Single Particle Analysis (SPA) is an ill-posed problem and so the determined structure can be the specular (mirrored) version of the true underlying structure. Macromolecules in nature have a specific handedness and it is important for atomic fitting of the structure that the resolved structure has the correct hand. At high resolution this is easily determined by looking at the map in a viewer like Chimera but it can be really difficult at resolutions of 4 to 5 Ångstroms. HaPi is a deep learning pipeline to automatically determine the hand of the electron density map automatically for structures below 5 Ångstroms of resolution.

=====
Cite
=====

J Garcia Condado, A. Muñoz-Barrutia, and C. O. S. Sorzano, “Automatic determination of the handedness of single-particle maps of macromolecules solved by CryoEM”, Journal of Structural Biology, vol. 214, no. 4, p. 107915, Dec. 2022. DOI: https://doi.org/10.1016/j.jsb.2022.107915

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

.. _Scipion: https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html

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

- Download PDBs:

Using the command line:

.. code-block:: shell

   chmod +x batch_download_pdb.sh 
   ./batch_download_pdb.sh -f {file_list_pdbs.txt} -o path/for/pdbs -p
   gunzip path/for/pdbs/*.pdb.gz

- Download Experimental cryoEM maps

Using the command line:

.. code-block:: shell

   chmod +x batch_download_emdb.sh 
   ./batch_download_emdb.sh -f {file_list_pdbs.txt} -o path/for/emdmaps
   
In **/data** you can find the following lists of PDBs and cyroEM maps (script it relates to from **/scripts**):

- ``emdb_list.txt`` (``test_pipeline_emdb_dataset.py``): List of all cryoEM maps accesible through EMDB as of August 2021
- ``EMDB_PDB_boxes.txt`` (``generate_experimental_boxes.py``): List of cryoEM maps used to train alpha helix 3DCNN
- ``EMDB_PDB_test.txt`` (``generate_exp_alpha_volumes.py`` >> ``test_alphavolnet_experimental_data.py``): List of cryoEM maps to test accuracy of alpha helix 3DCNN
- ``EMDB_hand_examples.txt`` (``test_pipeline_experimental_data.py``): List of cryoEM maps for visual testing
- ``PDB_boxes_list.txt`` (``generate_simulated_boxes.py``): List of PDBs used to simulate boxes for training
- ``PDB_volumes_list.txt`` (``generate_simulated_volumes.py`` >> ``test_pipeline_simulated_data.py``): List of PDBs used to simulate volumes


=========
Usage
=========

For training the models run (change direcotries in scripts and variables):

.. code-block:: shell

    scripts/generate_{simulated/experimental}_boxes.py
    scripts/create_torchdataset_{SSE/hand}.py
    scripts/train_{SSE/hand}.py
    scripts/test_model.py
    
For running the pipeline see ``scripts/test_pipeline_experimental_data.py``.

Use directly in Scipion_ by installing the plugin ``scipion-em-xmipp`` and using the protocol ``deep hand``.
