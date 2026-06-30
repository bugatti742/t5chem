# T5Chem MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

T5Chem MCP Server provides AI-powered chemical reaction predictions through the Model Context Protocol (MCP). This enables AI assistants to perform retrosynthesis analysis, product prediction, and molecular property calculations.

## Features

- **Retrosynthesis Prediction**: Predict reactants from a product molecule
- **Product Prediction**: Predict products from reactants and reagents
- **Reagents Prediction**: Predict required reagents for a reaction
- **Molecule Validation**: Validate SMILES strings
- **Molecular Properties**: Calculate detailed molecular properties (MW, LogP, TPSA, etc.)

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

Before using the MCP server, you need to download a pre-trained model:

```bash
# Download USPTO multi-task model
wget https://yzhang.hpc.nyu.edu/T5Chem/models/USPTO_MT_model.tar.bz2
tar -xjvf USPTO_MT_model.tar.bz2
```

## Quick Start with MCP

### Method 1: Project-level Configuration (Recommended)

The project already includes a `.mcp.json` configuration file. Simply clone and use:

```bash
git clone https://github.com/bugatti742/t5chem.git
cd t5chem
pip install -e ".[mcp]"

# Download model
wget https://yzhang.hpc.nyu.edu/T5Chem/models/USPTO_MT_model.tar.bz2
tar -xjvf USPTO_MT_model.tar.bz2 -C model/

# Open in your IDE - MCP server will be automatically detected!
```

### Method 2: Manual Configuration

Start the MCP server manually:

```bash
# Using default model path (model/)
t5chem-mcp

# Specify custom model path
t5chem-mcp --model_dir /path/to/your/model

# Or run as module
python -m t5chem.mcp_server --model_dir model/
```

## MCP Configuration Files

### Project-level Configuration (.mcp.json)

The project includes a `.mcp.json` file that allows IDEs to automatically detect and configure the MCP server:

```json
{
  "mcpServers": {
    "t5chem": {
      "command": "python",
      "args": ["-m", "t5chem.mcp_server", "--model_dir", "./model"],
      "cwd": "${workspaceFolder}",
      "env": {
        "T5CHEM_MODEL_DIR": "./model",
        "PYTHONPATH": "${workspaceFolder}"
      },
      "type": "stdio"
    }
  }
}
```

**Configuration Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `command` | The command to run the server | Yes |
| `args` | Command arguments | Yes |
| `cwd` | Working directory (use `${workspaceFolder}` for project root) | Yes |
| `env` | Environment variables | No |
| `type` | Connection type (`stdio` or `http`) | No (defaults to `stdio`) |

### Global Configuration

For global access across all projects, configure in your IDE's global settings:

**Claude Desktop**:
```json
{
  "mcpServers": {
    "t5chem": {
      "command": "python",
      "args": ["-m", "t5chem.mcp_server", "--model_dir", "/path/to/t5chem/model"],
      "cwd": "/path/to/t5chem"
    }
  }
}
```

**Cursor IDE**:
```json
{
  "mcp.servers": {
    "t5chem": {
      "command": "python",
      "args": ["-m", "t5chem.mcp_server", "--model_dir", "/path/to/model"],
      "cwd": "/path/to/t5chem"
    }
  }
}
```

## IDE Integration Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/bugatti742/t5chem.git
cd t5chem
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install with MCP support
pip install -e ".[mcp]"
```

### Step 3: Download Pre-trained Model

```bash
mkdir -p model
wget https://yzhang.hpc.nyu.edu/T5Chem/models/USPTO_MT_model.tar.bz2
tar -xjvf USPTO_MT_model.tar.bz2 -C model/
```

### Step 4: Open in IDE

**VS Code / Cursor**:
1. Open the `t5chem` folder
2. The `.mcp.json` file will be automatically detected
3. MCP server will connect automatically

**Claude Desktop**:
1. Open the `t5chem` folder
2. Go to Settings → MCP Servers
3. Click "Add Server" and paste the configuration from `mcp_config.example.json`

**Trae IDE**:
1. Open the `t5chem` folder
2. The `.mcp.json` file will be automatically loaded
3. MCP tools will be available in the AI assistant

### Step 5: Verify Connection

Check that the MCP server is connected:
- **VS Code/Cursor**: Look for "MCP: t5chem Connected" in the status bar
- **Claude Desktop**: Check MCP Servers list
- **Trae**: Check MCP tools are available in the tool list

## Available MCP Tools

Once connected, AI assistants can use these tools:

1. **predict_retrosynthesis**: Predict retrosynthesis routes for a target molecule
   - Input: `product_smiles` (string), `num_beams` (int, default: 10), `num_preds` (int, default: 5)
   - Output: list of predicted reactant combinations

2. **predict_product**: Predict product from reactants and reagents
   - Input: `reactants_smiles` (string), `reagents_smiles` (string, optional), `num_beams` (int), `num_preds` (int)
   - Output: list of predicted product SMILES

3. **predict_reagents**: Predict reagents needed for a reaction
   - Input: `reactants_smiles` (string), `product_smiles` (string), `num_beams` (int), `num_preds` (int)
   - Output: list of predicted reagents SMILES

4. **validate_molecule**: Validate a SMILES string and get basic info
   - Input: `smiles` (string)
   - Output: validity, canonical SMILES, molecular weight, formula, etc.

5. **get_molecule_properties**: Get detailed molecular properties
   - Input: `smiles` (string)
   - Output: MW, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, rings, etc.

## Example Usage

### Retrosynthesis Prediction

```python
# AI assistant will automatically call:
predict_retrosynthesis(
    product_smiles="COC(=O)c1nc(-c2cc([N+](=O)[O-])c(Cl)cc2F)cc(N)c1Cl",
    num_beams=10,
    num_preds=5
)
```

**Response:**
```json
{
  "product": "COC(=O)c1nc(-c2cc([N+](=O)[O-])c(Cl)cc2F)cc(N)c1Cl",
  "predictions": [
    "COC(=O)c1nc(-c2ccc(Cl)cc2F)cc(N)c1Cl.O=[N+]([O-])[O-]",
    "COC(=O)c1nc(-c2ccc(Cl)cc2F)cc(NC(C)=O)c1Cl.O=[N+]([O-])[O-]",
    "COC(=O)c1nc(Cl)cc(N)c1Cl.O=[N+]([O-])c1cc(Cl)cc(F)c1Cl",
    "COC(=O)c1nc(Cl)cc(N)c1Cl.O=[N+]([O-])c1cc(Cl)cc(F)c1",
    "COC(=O)c1nc(-c2ccc(F)cc2Cl)cc(N)c1Cl.O=[N+]([O-])[O-]"
  ],
  "num_predictions": 5,
  "model_dir": "./model"
}
```

### Molecular Properties

```python
get_molecule_properties(smiles="CC(=O)Oc1ccccc1C(=O)O")
```

**Response:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "formula": "C9H8O4",
  "molecular_weight": 180.16,
  "exact_mass": 180.0477,
  "logP": 1.19,
  "num_h_acceptors": 4,
  "num_h_donors": 1,
  "num_rotatable_bonds": 4,
  "num_rings": 1,
  "num_aromatic_rings": 1,
  "tpsa": 63.6
}
```

### Command Line (Legacy)

The original CLI is still available:

```bash
# Batch prediction
t5chem predict --data_dir data/sample/reactants/ --model_dir model/ --prediction predictions_res --batch_size 8

# Training
t5chem train --data_dir data/sample/reactants/ --output_dir model/ --task_type reactants
```

## Project Structure

```
t5chem/
├── t5chem/
│   ├── mcp_server.py       # MCP Server implementation (async)
│   ├── run_prediction.py   # Batch prediction
│   ├── run_trainer.py      # Model training
│   ├── model.py            # Model architecture
│   ├── data_utils.py       # Data utilities
│   └── mol_tokenizers.py   # Molecular tokenizers
├── model/                  # Trained model (download separately)
├── data/                   # Datasets (download separately)
├── .mcp.json               # Project-level MCP configuration (auto-detect)
├── mcp_config.example.json # Configuration template for IDEs
└── pyproject.toml          # Project configuration
```

## Environment Variables

- `T5CHEM_MODEL_DIR`: Path to the model directory (default: `model/`)
- `PYTHONPATH`: Add project root for module imports

## Requirements

- Python 3.10+
- PyTorch 2.2+
- Transformers 4.38+
- RDKit 2022.9+
- MCP SDK 1.0+

## Troubleshooting

### MCP Server Not Connecting

1. **Check Python path**: Ensure Python is in your PATH and using the correct virtual environment
2. **Verify model directory**: Ensure the model directory exists and contains model files
3. **Check dependencies**: Run `pip install -e ".[mcp]"` to ensure all dependencies are installed
4. **Test manually**: Run `t5chem-mcp` in terminal to see if there are any errors

### SMILES Validation Errors

- Ensure SMILES strings are valid chemical representations
- Try canonicalizing SMILES using RDKit before prediction
- Avoid invalid characters or malformed SMILES

### Performance Issues

- Use GPU for faster predictions (set `CUDA_VISIBLE_DEVICES`)
- Reduce `num_beams` for faster but less accurate predictions
- Increase batch size for batch prediction

## Citation

If you use T5Chem in your research, please cite:

```
Jieyu Lu and Yingkai Zhang, J Chem Inf Model, 62, 1376 - 1387 (2022)
https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c01467
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Bug Reports

If you encounter any bugs, please report the issue at:
https://github.com/bugatti742/t5chem/issues