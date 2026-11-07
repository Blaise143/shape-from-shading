Usage
=====

Installation
------------

.. code-block:: bash

   conda create -n sfs python=3.11
   conda activate sfs
   pip install -r requirements.txt

Data layout
-----------

To run this, you will need:

- an **image directory** with grayscale images
- a **mask directory** with binary masks indicating the objects of interest
- image and mask **basenames must match**, e.g. ``images/sphere.png`` and
  ``masks/sphere.png``

Basic usage
-----------

Run over all objects found in the image directory:

.. code-block:: bash

   python main.py --images-dir data/images --masks-dir data/masks --out-path out

Run on a single object (for instance, a sphere):

.. code-block:: bash

   python main.py --images-dir data/images --masks-dir data/masks --out-path out --target sphere

Optional arguments
-------------------

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
   * - ``--num-iters``
     - 2000
   * - ``--lr``
     - 0.01
   * - ``--lambda-smooth``
     - 1.0
   * - ``--light-direction LX LY LZ``
     - 0.0 0.0 1.0
   * - ``--no-optimize-light`` / ``--optimize-light``
     - optimize-light=True
   * - ``--optimize-albedo``
     - disabled
   * - ``--init-kind``
     - init_grads (``init_grads``, ``zeroes``, ``ones``, ``sphere``)
   * - ``--parametrization-type``
     - stereographic (``cartesian``, ``stereographic``)
   * - ``--use-wandb``
     - disabled
   * - ``--log-intermediates``
     - disabled

See :doc:`api` for the full programmatic interface.
