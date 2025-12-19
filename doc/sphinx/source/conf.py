# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from regpy._version import version_tuple

project = 'RegPy'
copyright = '2024, Thorsten Hohage'
author = 'Thorsten Hohage'
version = f"{version_tuple[0]}.{version_tuple[1]}"
release = f"{version_tuple[0]}.{version_tuple[1]}.{version_tuple[2]}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'sphinx.ext.intersphinx',
    "sphinx.ext.viewcode",
    "nbsphinx",
    # "sphinx_mdinclude",
    "myst_parser",
    "sphinx.ext.mathjax",
    ]

templates_path = ['_templates']
exclude_patterns = []

autoapi_dirs = ['../../../regpy']
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_add_toctree_entry = False
# uncomment to determine bugs
# autoapi_keep_files = True

nbsphinx_allow_errors = True

viewcode_follow_imported_members = True

autodoc_typehints = "signature"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

master_doc = "contents"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'furo'
html_static_path = ['_static']
