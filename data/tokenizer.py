# my_llm_project/data/tokenizer.py
import json
import os
from .. import config # Import global config

class SimpleCharTokenizer:
    def __init__(self, corpus=None, vocab_path=None):
        self.vocab = {}
        self.inv_vocab = {}
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>" # For unknown characters

        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        elif corpus:
            self._build_vocab(corpus)
        else:
            # Initialize with special tokens even if no corpus/path
            self._add_special_tokens()


        # Update global config with actual token IDs and vocab size
        config.PAD_TOKEN_ID = self.vocab.get(self.pad_token, 0) # Default to 0 if not found
        config.BOS_TOKEN_ID = self.vocab.get(self.bos_token, 1)
        config.EOS_TOKEN_ID = self.vocab.get(self.eos_token, 2)
        config.UNK_TOKEN_ID = self.vocab.get(self.unk_token, 3)
        config.VOCAB_SIZE = len(self.vocab)


    def _add_special_tokens(self):
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token

    def _build_vocab(self, corpus_list):
        self._add_special_tokens()
        chars = set()
        for text in corpus_list:
            chars.update(list(text))
        
        for char in sorted(list(chars)):
            if char not in self.vocab: # Avoid re-adding if already a special token name
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inv_vocab[idx] = char

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[self.pad_token]

    @property
    def bos_id(self):
        return self.vocab[self.bos_token]

    @property
    def eos_id(self):
        return self.vocab[self.eos_token]
    
    @property
    def unk_id(self):
        return self.vocab[self.unk_token]

    def encode(self, text, add_special_tokens=True):
        tokens = [self.vocab.get(char, self.unk_id) for char in text]
        if add_special_tokens:
            return [self.bos_id] + tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        chars = []
        for token_id in token_ids:
            token = self.inv_vocab.get(token_id)
            if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            if token == self.unk_token and skip_special_tokens: # Optionally skip UNK too
                continue
            if token:
                chars.append(token)
        return "".join(chars)

    def save_vocab(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab, 'inv_vocab': self.inv_vocab}, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {filepath}")

    def load_vocab(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            # inv_vocab keys might be strings after JSON load, convert them to int
            self.inv_vocab = {int(k): v for k, v in data['inv_vocab'].items()}
        print(f"Vocabulary loaded from {filepath}")
        # Ensure special tokens are set correctly from loaded vocab
        self.pad_token = self.inv_vocab.get(0, "<PAD>") # Assuming 0 is PAD
        self.bos_token = self.inv_vocab.get(1, "<BOS>") # Assuming 1 is BOS
        self.eos_token = self.inv_vocab.get(2, "<EOS>") # Assuming 2 is EOS
        self.unk_token = self.inv_vocab.get(3, "<UNK>") # Assuming 3 is UNK