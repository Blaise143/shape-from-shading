# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Shape-from-Shading"
copyright = "2026, Blaise Appolinary"
author = "Blaise Appolinary"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Heavy/third-party deps aren't needed to render docstrings and would make
# the docs build slow (or fail) on Read the Docs, so autodoc imports mocks
# of them instead of the real packages.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "wandb",
    "tqdm",
    "plotly",
    "scipy",
    "matplotlib",
    "PIL",
    "imageio",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}
html_static_path = ["_static"]
