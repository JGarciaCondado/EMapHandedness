==========================
HaPi - Handedness Pipeline
==========================



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

====
Data
====

To download the data used a set of batch-downlaod shell scripts and the PDB and EMD IDs are available in **/data**.

- Download PDBs used to simulate boxes

Using the command line:

.. code-block:: shell

   chmod +x batch_download_pdb.sh 
   ./batch_download_pdb.sh -f PDB_boxes_list.txt -o path/for/pdbs -p
   gunzip path/for/pdbs/*.pdb.gz

- Download PDBS used to simulate volumes

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
