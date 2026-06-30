import argparse
import os
import sys

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from transformers import T5Config, T5ForConditionalGeneration

from t5chem.mol_tokenizers import AtomTokenizer, SelfiesTokenizer, SimpleTokenizer

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict retrosynthesis for a single molecule")
    parser.add_argument("smiles", type=str, help="SMILES string of the target molecule")
    parser.add_argument("--model_dir", type=str, default="model/", help="Path to the model directory")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams for beam search")
    parser.add_argument("--num_preds", type=int, default=5, help="Number of predictions to return")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = T5Config.from_pretrained(args.model_dir)
    tokenizer_type = getattr(config, "tokenizer", "simple")
    
    if tokenizer_type == "simple":
        Tokenizer = SimpleTokenizer
    elif tokenizer_type == 'atom':
        Tokenizer = AtomTokenizer
    else:
        Tokenizer = SelfiesTokenizer
    
    tokenizer = Tokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.txt'))
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()
    model = model.to(device)
    
    task_specific_params = {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_preds,
        "decoder_start_token_id": tokenizer.pad_token_id,
    }
    
    input_text = "Reactants:" + args.smiles
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            **task_specific_params
        )
    
    print(f"Target molecule: {args.smiles}")
    print(f"\nPredictions ({args.num_preds}):")
    for i, pred in enumerate(outputs, 1):
        pred_smiles = tokenizer.decode(
            pred,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).replace(" ","")
        print(f"  {i}. {pred_smiles}")

if __name__ == "__main__":
    main()