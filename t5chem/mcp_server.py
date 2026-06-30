#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5Chem MCP Server - Model Context Protocol Server for Chemical Reaction Predictions
"""

import asyncio
import os
import logging
from typing import Optional

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from mcp.server.fastmcp import FastMCP
from transformers import T5Config, T5ForConditionalGeneration

from t5chem.mol_tokenizers import AtomTokenizer, SelfiesTokenizer, SimpleTokenizer

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("T5CHEM_MODEL_DIR", "model/")

mcp = FastMCP(
    "t5chem",
    instructions="T5Chem MCP Server - AI-powered chemical reaction prediction and retrosynthesis analysis",
)

_model_cache = None
_tokenizer_cache = None
_device_cache = None


async def load_model_and_tokenizer(model_dir: str):
    global _model_cache, _tokenizer_cache, _device_cache
    
    if _model_cache is not None:
        return _model_cache, _tokenizer_cache, _device_cache
    
    def _load():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from: {model_dir}")
        logger.info(f"Using device: {device}")
        
        config = T5Config.from_pretrained(model_dir)
        tokenizer_type = getattr(config, "tokenizer", "simple")
        
        if tokenizer_type == "simple":
            Tokenizer = SimpleTokenizer
        elif tokenizer_type == 'atom':
            Tokenizer = AtomTokenizer
        else:
            Tokenizer = SelfiesTokenizer
        
        tokenizer = Tokenizer(vocab_file=os.path.join(model_dir, 'vocab.txt'))
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        model.eval()
        model = model.to(device)
        
        return model, tokenizer, device
    
    model, tokenizer, device = await asyncio.to_thread(_load)
    
    _model_cache = model
    _tokenizer_cache = tokenizer
    _device_cache = device
    
    logger.info("Model loaded successfully")
    return model, tokenizer, device


def validate_smiles(smiles: str) -> tuple[bool, str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES: {smiles}"
        return True, ""
    except Exception as e:
        return False, f"Error validating SMILES: {str(e)}"


@mcp.tool()
async def predict_retrosynthesis(
    product_smiles: str,
    num_beams: int = 10,
    num_preds: int = 5,
    model_dir: Optional[str] = None
) -> dict:
    if num_preds > num_beams:
        raise ValueError("num_preds must be <= num_beams")
    
    is_valid, error_msg = validate_smiles(product_smiles)
    if not is_valid:
        raise ValueError(error_msg)
    
    model_path = model_dir or MODEL_DIR
    model, tokenizer, device = await load_model_and_tokenizer(model_path)
    
    def _generate():
        task_specific_params = {
            "early_stopping": True,
            "max_length": 300,
            "num_beams": num_beams,
            "num_return_sequences": num_preds,
            "decoder_start_token_id": tokenizer.pad_token_id,
        }
        
        input_text = "Reactants:" + product_smiles
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                **task_specific_params
            )
        
        predictions = []
        for pred in outputs:
            pred_smiles = tokenizer.decode(
                pred,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).replace(" ","")
            predictions.append(pred_smiles)
        
        return predictions
    
    predictions = await asyncio.to_thread(_generate)
    
    return {
        "product": product_smiles,
        "predictions": predictions,
        "num_predictions": len(predictions),
        "model_dir": model_path
    }


@mcp.tool()
async def predict_product(
    reactants_smiles: str,
    reagents_smiles: str = "",
    num_beams: int = 10,
    num_preds: int = 5,
    model_dir: Optional[str] = None
) -> dict:
    if num_preds > num_beams:
        raise ValueError("num_preds must be <= num_beams")
    
    is_valid, error_msg = validate_smiles(reactants_smiles)
    if not is_valid:
        raise ValueError(f"Invalid reactants SMILES: {error_msg}")
    
    if reagents_smiles:
        is_valid, error_msg = validate_smiles(reagents_smiles)
        if not is_valid:
            raise ValueError(f"Invalid reagents SMILES: {error_msg}")
    
    input_smiles = f"{reactants_smiles}>>{reagents_smiles}"
    model_path = model_dir or MODEL_DIR
    model, tokenizer, device = await load_model_and_tokenizer(model_path)
    
    def _generate():
        task_specific_params = {
            "early_stopping": True,
            "max_length": 200,
            "num_beams": num_beams,
            "num_return_sequences": num_preds,
            "decoder_start_token_id": tokenizer.pad_token_id,
        }
        
        input_text = "Product:" + input_smiles
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                **task_specific_params
            )
        
        predictions = []
        for pred in outputs:
            pred_smiles = tokenizer.decode(
                pred,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).replace(" ","")
            predictions.append(pred_smiles)
        
        return predictions
    
    predictions = await asyncio.to_thread(_generate)
    
    return {
        "reactants": reactants_smiles,
        "reagents": reagents_smiles,
        "predictions": predictions,
        "num_predictions": len(predictions),
        "model_dir": model_path
    }


@mcp.tool()
async def predict_reagents(
    reactants_smiles: str,
    product_smiles: str,
    num_beams: int = 10,
    num_preds: int = 5,
    model_dir: Optional[str] = None
) -> dict:
    if num_preds > num_beams:
        raise ValueError("num_preds must be <= num_beams")
    
    is_valid, error_msg = validate_smiles(reactants_smiles)
    if not is_valid:
        raise ValueError(f"Invalid reactants SMILES: {error_msg}")
    
    is_valid, error_msg = validate_smiles(product_smiles)
    if not is_valid:
        raise ValueError(f"Invalid product SMILES: {error_msg}")
    
    input_smiles = f"{reactants_smiles}>>{product_smiles}"
    model_path = model_dir or MODEL_DIR
    model, tokenizer, device = await load_model_and_tokenizer(model_path)
    
    def _generate():
        task_specific_params = {
            "early_stopping": True,
            "max_length": 200,
            "num_beams": num_beams,
            "num_return_sequences": num_preds,
            "decoder_start_token_id": tokenizer.pad_token_id,
        }
        
        input_text = "Reagents:" + input_smiles
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                **task_specific_params
            )
        
        predictions = []
        for pred in outputs:
            pred_smiles = tokenizer.decode(
                pred,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).replace(" ","")
            predictions.append(pred_smiles)
        
        return predictions
    
    predictions = await asyncio.to_thread(_generate)
    
    return {
        "reactants": reactants_smiles,
        "product": product_smiles,
        "predictions": predictions,
        "num_predictions": len(predictions),
        "model_dir": model_path
    }


@mcp.tool()
async def validate_molecule(smiles: str) -> dict:
    is_valid, error_msg = validate_smiles(smiles)
    
    if not is_valid:
        return {
            "valid": False,
            "input_smiles": smiles,
            "error": error_msg
        }
    
    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol)
    molecular_weight = Descriptors.MolWt(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    
    return {
        "valid": True,
        "input_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "molecular_weight": round(molecular_weight, 2),
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
        "formula": formula
    }


@mcp.tool()
async def get_molecule_properties(smiles: str) -> dict:
    is_valid, error_msg = validate_smiles(smiles)
    if not is_valid:
        raise ValueError(error_msg)
    
    mol = Chem.MolFromSmiles(smiles)
    
    return {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 2),
        "logP": round(Descriptors.MolLogP(mol), 2),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_rings": Descriptors.RingCount(mol),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2)
    }


def main():
    import argparse
    
    global MODEL_DIR
    
    parser = argparse.ArgumentParser(description="T5Chem MCP Server")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=MODEL_DIR,
        help="Path to the trained model directory"
    )
    args = parser.parse_args()
    
    MODEL_DIR = args.model_dir
    
    logger.info(f"Starting T5Chem MCP Server with model: {MODEL_DIR}")
    mcp.run()


if __name__ == "__main__":
    main()