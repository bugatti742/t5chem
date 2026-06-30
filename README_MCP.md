# T5Chem MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

T5Chem MCP Server provides AI-powered chemical reaction predictions through the Model Context Protocol (MCP).

## Features

- **Retrosynthesis Prediction**: Predict reactants from a product molecule
- **Product Prediction**: Predict products from reactants and reagents
- **Reagents Prediction**: Predict required reagents for a reaction
- **Molecule Validation**: Validate SMILES strings
- **Molecular Properties**: Calculate detailed molecular properties

## Installation

```bash
git clone https://github.com/bugatti742/t5chem.git
cd t5chem
pip install -e ".[mcp]"
```

## Download Pre-trained Model

```bash
wget https://yzhang.hpc.nyu.edu/T5Chem/models/USPTO_MT_model.tar.bz2
tar -xjvf USPTO_MT_model.tar.bz2 -C model/
```

## Usage

### Start MCP Server

```bash
t5chem-mcp --model_dir model/
```

### Available MCP Tools

1. **predict_retrosynthesis**: Predict retrosynthesis routes
2. **predict_product**: Predict product from reactants
3. **predict_reagents**: Predict reagents for a reaction
4. **validate_molecule**: Validate SMILES strings
5. **get_molecule_properties**: Get molecular properties

### Example

```python
from t5chem.mcp_server import predict_retrosynthesis

result = predict_retrosynthesis(
    product_smiles="CC(=O)c1ccccc1",
    num_beams=10,
    num_preds=5
)
print(result["predictions"])
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- Transformers 4.38+
- RDKit 2022.9+
- MCP SDK 1.0+

## License

MIT License