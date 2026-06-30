import importlib
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Union, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

is_selfies_available: bool = False
if importlib.util.find_spec("selfies"):
    from selfies import split_selfies
    is_selfies_available = True
pattern: str = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex: re.Pattern = re.compile(pattern)
TASK_PREFIX: List[str] = ['Yield:', 'Product:', 'Fill-Mask:', 'Classification:', 'Reagents:', 'Reactants:']


class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = tokens
        self.stoi = {s: i for i, s in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for token in self.itos:
                f.write(f"{token}\n")

    @classmethod
    def load(cls, path: str) -> 'Vocab':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        with open(path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f]
        return cls(tokens)


class MolTokenizer(ABC, PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        unk_token: str='<unk>',
        bos_token: str='<s>',
        pad_token: str="<pad>",
        eos_token: str='</s>',
        mask_token: str='<mask>',
        max_size: int=1000,
        task_prefixs: List[str]=[],
        **kwargs
    ) -> None:
        task_prefixs = TASK_PREFIX+task_prefixs
        self.create_vocab(vocab_file=vocab_file)
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs)
        if self.vocab:
            extra_to_add: int = max_size - len(self.vocab)
            cur_added_len: int = len(task_prefixs) + 9
            for i in range(cur_added_len, extra_to_add):
                task_prefixs.append('<extra_task_{}>'.format(str(i)))
            self.add_tokens(['<extra_token_'+str(i)+'>' for i in range(9)]+task_prefixs+['>'], special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def create_vocab(self, vocab_file: Optional[str]=None) -> None:
        self.vocab: Vocab = Vocab.load(vocab_file)

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    @abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        pass

    def _convert_token_to_id(self, token: str) -> int:
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index: int) -> str:
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        out_string: str = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1
        
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory")

        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
        )
        
        self.vocab.save(vocab_file)
        return (vocab_file,)


class SimpleTokenizer(MolTokenizer):
    def __init__(self, vocab_file, max_size=100, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        return list(text)


class AtomTokenizer(MolTokenizer):
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        tokens: List[str] = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens

class SelfiesTokenizer(MolTokenizer):
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)
        assert is_selfies_available, "You need to install selfies package to use SelfiesTokenizer"

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        return list(split_selfies(text))