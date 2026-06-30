T5Chem
=======

A Unified Deep Learning Model for Multi-task Reaction Predictions

Features
--------

- **Retrosynthesis Prediction**: Predict reactants from a product molecule
- **Product Prediction**: Predict products from reactants and reagents
- **Reagents Prediction**: Predict required reagents for a reaction
- **Classification**: Predict reaction classes
- **Regression**: Predict reaction yields
- **MCP Support**: Model Context Protocol integration for AI assistants

Installation
------------

.. code:: bash

    pip install -e .

Quick Start
-----------

.. code:: bash

    t5chem predict --data_dir data/sample/reactants/ --model_dir model/

For more details, see the documentation.

Requirements
------------

- Python >= 3.10
- PyTorch >= 2.2.0
- transformers >= 4.38.0
- rdkit >= 2022.9.4

License
-------

MIT License

Other projects in Zhang's Lab:
https://www.nyu.edu/projects/yzhang/IMA/

MCP Server Support
------------------

T5Chem now supports the Model Context Protocol (MCP) for AI assistants.

.. code:: bash

   # Install with MCP support
   pip install -e ".[mcp]"
   
   # Start MCP server
   t5chem-mcp --model_dir model/

Download Model and Data
-----------------------

Large files are NOT included in the GitHub repository.
Download them separately:

- Pre-trained models: Download from Zenodo or T5Chem website
- Extract to model/ directory

Known Issues
------------

1. Starlette Version Conflict: The MCP SDK requires starlette>=1.3.1
2. Model Files Not in Git: Model weights excluded from version control
3. GPU Acceleration: Recommended for faster predictions

Troubleshooting
---------------

- Verify SMILES format correctness
- Check model_dir points to valid directory
- Use GPU for better performance