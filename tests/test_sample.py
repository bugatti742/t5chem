import pytest
from t5chem.mol_tokenizers import SimpleTokenizer
from t5chem.data_utils import T5ChemTasks

def test_tokenizer():
    tokenizer = SimpleTokenizer(vocab_file='t5chem/vocab/simple.txt')
    assert tokenizer.vocab_size > 0
    tokens = tokenizer._tokenize('CC(=O)c1ccccc1')
    assert len(tokens) > 0

def test_task_settings():
    assert 'reactants' in T5ChemTasks
    assert 'product' in T5ChemTasks
    assert 'reagents' in T5ChemTasks
    task = T5ChemTasks['reactants']
    assert task.prefix == 'Reactants:'