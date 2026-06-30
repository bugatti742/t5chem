# T5Chem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Unified Deep Learning Model for Multi-task Reaction Predictions with MCP (Model Context Protocol) support.

## Features

- **Retrosynthesis Prediction**: Predict reactants from a product molecule
- **Product Prediction**: Predict products from reactants and reagents
- **Reagents Prediction**: Predict required reagents for a reaction
- **Molecule Validation**: Validate SMILES strings
- **Molecular Properties**: Calculate detailed molecular properties
- **MCP Server**: Integrate with AI assistants through Model Context Protocol

## Installation

```bash
# Clone the repository
git clone https://github.com/bugatti742/t5chem.git
cd t5chem

# Install with MCP support
pip install -e ".[mcp]"

# Or install all dependencies
pip install -e .
```

## Download Pre-trained Model

Large model files are NOT included in the repository. Download them separately:

```bash
# Download USPTO multi-task model
wget https://yzhang.hpc.nyu.edu/T5Chem/models/USPTO_MT_model.tar.bz2
tar -xjvf USPTO_MT_model.tar.bz2 -C model/
```

## Usage

### As MCP Server

Start the MCP server:

```bash
# Using default model path (model/)
t5chem-mcp

# Specify custom model path
t5chem-mcp --model_dir /path/to/your/model
```

### Available MCP Tools

1. **predict_retrosynthesis**: Predict retrosynthesis routes
2. **predict_product**: Predict product from reactants
3. **predict_reagents**: Predict reagents for a reaction
4. **validate_molecule**: Validate SMILES strings
5. **get_molecule_properties**: Get molecular properties

### Command Line

```bash
# Batch prediction
t5chem predict --data_dir data/sample/reactants/ --model_dir model/

# Training
t5chem train --data_dir data/sample/reactants/ --output_dir model/ --task_type reactants
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- Transformers 4.38+
- RDKit 2022.9+
- MCP SDK 1.0+

## Citation

Jieyu Lu and Yingkai Zhang, J Chem Inf Model, 62, 1376 - 1387 (2022)

## License

MIT License