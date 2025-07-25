# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

now = datetime.now()

project = "padre_meddea"
copyright = f"US Government (not copyrighted) 2023-{now.year}"
author = "The PADRE MeDDEA Team"

from padre_meddea import __version__

release = __version__
is_development = ".dev" in __version__
# -- Project information -----------------------------------------------------



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
]

# to exclude traditional Python prompts from your copied code
copybutton_prompt_text = ">>> "

# plot_directive default to always show source when including a plot
plot_include_source = True

# Set automodapi to generate files inside the generated directory
# automodapi_toctreedirnm = "_build/api"
autosummary_generate = True  # Turn on sphinx.ext.autosummary
numpydoc_show_class_members = False
# generate autosummary even if no references
autosummary_generate = True

autosummary_ignore_module_all = False
autosummary_imported_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = "obj"

# Disable having a separate return type row
napoleon_use_rtype = False

# Disable google style docstrings
napoleon_google_docstring = False

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://data.astropy.org/intersphinx/python3.inv"),
    ),
    "numpy": (
        "https://docs.scipy.org/doc/numpy/",
        (None, "https://numpy.org/doc/stable/objects.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        (None, "https://docs.scipy.org/doc/scipy/objects.inv"),
    ),
    "matplotlib": (
        "https://matplotlib.org/",
        (None, "https://matplotlib.org/stable/objects.inv"),
    ),
    "astropy": (
        "http://docs.astropy.org/en/stable/",
        "https://docs.astropy.org/en/stable/objects.inv",
    ),
    "sunpy": (
        "https://docs.sunpy.org/en/stable/",
        "https://docs.sunpy.org/en/stable/objects.inv",
    ),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "bizstyle"
html_static_path = ["_static"]

html_logo = "logo/padre_logo.png"
# html_favicon = "logo/favicon.ico"
html_css_files = ["css/custom.css"]
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]
