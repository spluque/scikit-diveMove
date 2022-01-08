# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from skdiveMove import __version__

sys.path.append("../skdiveMove")


# -- Project information -----------------------------------------------------

project = 'skdiveMove'
copyright = '2022, Sebastian Luque'
author = 'Sebastian Luque'

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'jupyter_sphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['.templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# jupyter_sphinx
jupyter_sphinx_linenos = True
jupyter_sphinx_continue_linenos = True

# autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'groupwise'
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinxdoc'
html_theme = 'bizstyle'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {
#     'stickysidebar': 'true',
#     'sidebarbgcolor': "#3E3B32",
#     # 'relbarbgcolor': "#19205E",
#     # 'footerbgcolor': "#19205E",
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['.static']

# Name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or
# 32x32 pixels large.
html_logo = '.static/skdiveMove_logo.png'

# Output file base name for HTML help builder.
htmlhelp_basename = 'skdiveMove_doc'


# -- Options for LaTeX output ---------------------------------------------

# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     # 'papersize': 'letterpaper',

#     # The font size ('10pt', '11pt' or '12pt').
#     # 'pointsize': '10pt',

#     # Additional stuff for the LaTeX preamble.
#     # 'preamble': '',
# }

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto/manual]).
latex_documents = [
    (master_doc, 'index', 'skdiveMove.tex', u'skdiveMove Documentation',
     u'Sebastian Luque', 'manual'),
]

# -- Extension configurations ---------------------------------------------

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
}
