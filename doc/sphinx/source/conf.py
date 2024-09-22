# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RegPy'
copyright = '2024, Thorsten Hohage'
author = 'Thorsten Hohage'
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autoapi.extension',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_mdinclude"
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

nbsphinx_allow_errors = True

viewcode_follow_imported_members = True


autodoc_typehints = "signature"
# autoapi_keep_files = True

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

source_suffix = ['.rst', '.md']

html_theme = 'furo'
html_static_path = ['_static']
