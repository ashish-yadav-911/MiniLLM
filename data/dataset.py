# my_llm_project/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .. import config # Import global config

class TextDataset(Dataset):
    def __init__(self, tokenized_sequences):
        """
        Args:
            tokenized_sequences (list of lists of int): A list where each inner list
                                                        is a sequence of token IDs.
        """
        self.sequences = tokenized_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # For decoder-only, input is sequence[:-1], target is sequence[1:]
        full_seq = self.sequences[idx]
        
        # Ensure there's at least one token for target
        if len(full_seq) < 2:
            # This case should ideally be filtered out during preprocessing
            # Or handle by returning empty tensors / specific error
            print(f"Warning: Sequence at index {idx} is too short: {full_seq}")
            # Fallback: return the sequence itself and let collate_fn handle potential errors or padding
            # Or, better, ensure preprocessing filters these out.
            # For now, let's assume preprocessing ensures sequences are at least length 2.
            # If not, the slicing below will cause issues.
            # A robust way is to filter such short sequences before creating the Dataset.
            # For this example, we'll proceed assuming len(full_seq) >= 2
            pass

        input_ids = torch.tensor(full_seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(full_seq[1:], dtype=torch.long)
        return {"input_ids": input_ids, "target_ids": target_ids}

def collate_fn(batch, pad_token_id):
    """
    Pads sequences in a batch to the same length.
    Args:
        batch (list of dicts): Each dict has "input_ids" and "target_ids".
        pad_token_id (int): The ID of the padding token.
    Returns:
        dict: A dictionary with "input_ids" and "target_ids" as padded tensors.
    """
    input_ids_list = [item["input_ids"] for item in batch]
    target_ids_list = [item["target_ids"] for item in batch]

    # Pad sequences
    # batch_first=True means output shape is (batch_size, seq_len)
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    padded_target_ids = pad_sequence(target_ids_list, batch_first=True, padding_value=pad_token_id) # Use pad_token_id here, loss will ignore it

    return {
        "input_ids": padded_input_ids,
        "target_ids": padded_target_ids
    }

def prepare_data_for_lm(raw_texts, tokenizer, max_seq_len):
    """
    Tokenizes texts, chunks them, and prepares for language modeling.
    Each chunk will be `[BOS] token1 token2 ... tokenN [EOS]`.
    The dataset will then create input/target pairs from these full sequences.
    """
    all_tokenized_sequences = []
    for text in raw_texts:
        # Encode adds BOS and EOS by default if tokenizer is configured for it
        encoded_text = tokenizer.encode(text, add_special_tokens=True)

        # Chunking (optional, but good for long texts)
        # If not chunking, just ensure sequences are not too long or handle truncation
        # For simplicity, let's assume texts are reasonably short or we truncate.
        # A more robust approach would be to chunk long texts into max_seq_len segments.
        if len(encoded_text) > max_seq_len:
            encoded_text = encoded_text[:max_seq_len] # Simple truncation

        if len(encoded_text) >= 2: # Need at least BOS + one token for a valid input/target pair
            all_tokenized_sequences.append(encoded_text)
            
    return all_tokenized_sequences


def create_data_loaders(raw_train_texts, raw_val_texts, tokenizer, batch_size, max_seq_len, num_workers=0):
    train_tokenized = prepare_data_for_lm(raw_train_texts, tokenizer, max_seq_len)
    val_tokenized = prepare_data_for_lm(raw_val_texts, tokenizer, max_seq_len)

    train_dataset = TextDataset(train_tokenized)
    val_dataset = TextDataset(val_tokenized)

    pad_id = tokenizer.pad_id # Get pad_id from the tokenizer instance

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id), # Pass pad_id to collate_fn
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    return train_loader, val_loader