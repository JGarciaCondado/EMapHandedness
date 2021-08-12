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
