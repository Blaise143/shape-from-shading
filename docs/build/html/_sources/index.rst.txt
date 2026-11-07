Shape-from-Shading documentation
=================================

This repo includes a simple implementation of a Shape-from-Shading pipeline.
There are two variants of the optimization implemented:

- **Direct normal estimation:** the normal components :math:`(n_x, n_y, n_z)`
  are optimized directly.
- **Stereographic parametrization:** stereographic parameters are optimized
  and mapped back to normals.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
