from functools import partial
import tiktoken
import torch
from .datasets import GPTDatasetV1, SpamDataset, InstructionDataset
from torch.utils.data import DataLoader

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def create_dataloader(txt, batch_size=4,
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=1,
                         dataset_type="default",
                         device="cpu"):

    # Initialize the collate_fn to None

    customized_collate_fn = None

    # Create dataset
    if dataset_type == "default":
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    elif dataset_type == "spam":
        dataset = SpamDataset(txt, tokenizer, max_length, stride)
    elif dataset_type == "instruction":
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=max_length)
        dataset = InstructionDataset(txt, tokenizer)
    else:
        print("Unknown dataset")
        raise Exception("Unknown dataset type")


    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last)

    return dataloader

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor




