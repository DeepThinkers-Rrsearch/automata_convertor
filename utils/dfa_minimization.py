import os
import pickle
import torch
import torch.nn as nn

# device & hyper-parameters must match what you used at training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LEN = 200
EMBED_DIM = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.2

class RemappingUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == "__main__" and name == "CharTokenizer":
            return CharTokenizer
        return super().find_class(module, name)
    


class CharTokenizer:
    def __init__(self, texts):
        chars = sorted(set(''.join(texts)))
        self.stoi = {ch: i+3 for i, ch in enumerate(chars)}  # reserve 0,1,2
        # special tokens
        self.stoi['<pad>'] = 0
        self.stoi['<sos>'] = 1
        self.stoi['<eos>'] = 2
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s, max_len, add_special=True):
        ids = []
        if add_special:
            ids.append(self.stoi['<sos>'])
        ids += [self.stoi.get(ch, 0) for ch in s]
        if add_special:
            ids.append(self.stoi['<eos>'])
        ids = ids[:max_len]
        pad_len = max_len - len(ids)
        ids += [self.stoi['<pad>']] * pad_len
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            ch = self.itos.get(i, '')
            if ch == '<eos>': break
            if ch not in ['<pad>', '<sos>']:
                tokens.append(ch)
        return ''.join(tokens)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads,
                 num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(
            self._generate_positional_encoding(MAX_SEQ_LEN, embed_dim), requires_grad=False
        )
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def _generate_positional_encoding(self, max_len, d_model):
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        angle_rates = 1 / (10000 ** (i.float() / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * angle_rates)
        pe[:, 1::2] = torch.cos(pos * angle_rates)
        return pe.unsqueeze(0)

    def forward(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(output)
    
def load_tokenizer(tokenizer_path: str):
    """Load and return the pickled CharTokenizer."""
    with open(tokenizer_path, 'rb') as f:
         return RemappingUnpickler(f).load()

def load_dfa_minimization_model(model_path,tokenizer_path):
    tokenizer  = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.stoi)
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# def predict_dfa_minimization(model: Seq2SeqTransformer,
#             src_str: str,
#             max_len: int = MAX_SEQ_LEN) -> str:
#     """
#     Given a source string, returns the model’s decoded output string.
#     Uses greedy one‐token‐at‐a‐time decoding until <eos> or max_len.
#     """
#     # encode source
#     tokenizer  = load_tokenizer("models/dfa_minimization/dfa_minimizer_tokenizer.pkl")
#     src_ids = tokenizer.encode(src_str, max_len, add_special=True)
#     src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

#     # start target with <sos>
#     tgt_ids = [tokenizer.stoi['<sos>']]
#     with torch.no_grad():
#         for _ in range(max_len):
#             tgt = torch.tensor([tgt_ids], dtype=torch.long, device=DEVICE)
#             out = model(src, tgt)                   # (1, seq_len, vocab_size)
#             next_id = out.argmax(-1)[0, -1].item() # pick highest‐prob token
#             if next_id == tokenizer.stoi['<eos>']:
#                 break
#             tgt_ids.append(next_id)

#     # decode (skips <sos> and stops at <eos>)
#     return tokenizer.decode(tgt_ids)

def predict_dfa_minimization(model, src_str, max_len=MAX_SEQ_LEN):
    model.eval()
    tokenizer  = load_tokenizer("models/dfa_minimization/dfa_minimizer_tokenizer.pkl")
    src_ids = torch.tensor(tokenizer.encode(src_str, max_len, add_special=False)).unsqueeze(0).to(DEVICE)
    tgt_ids = torch.full((1, max_len), tokenizer.stoi['<pad>'], dtype=torch.long).to(DEVICE)
    tgt_ids[0,0] = tokenizer.stoi['<sos>']
    for i in range(1, max_len):
        with torch.no_grad():
            out = model(src_ids, tgt_ids[:,:i])
        next_token = out.argmax(-1)[0, i-1].item()
        tgt_ids[0,i] = next_token
        if next_token == tokenizer.stoi['<eos>']:
            break
    return tokenizer.decode(tgt_ids[0].cpu().tolist())