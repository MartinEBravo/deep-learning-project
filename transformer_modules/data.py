import torch

def split_into_blocks(encoded_data, block_size, pad_token_id):
    num_blocks = (len(encoded_data) + block_size - 1) // block_size
    blocks = [encoded_data[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]
    blocks = [torch.cat([b, torch.full((block_size - len(b),), pad_token_id)])
                if len(b) < block_size else b for b in blocks]
    return blocks

def load_data(input_file, encode, block_size, pad_token_id):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    encoded_data = torch.tensor(encode(text), dtype=torch.long)
    blocks = split_into_blocks(encoded_data, block_size, pad_token_id)
    data = torch.cat(blocks)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)