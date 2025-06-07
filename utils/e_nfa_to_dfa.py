import torch
import torch.nn as nn
import pickle

EMBED_SIZE = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
FFN_HIDDEN_DIM = 512
DROPOUT_RATE = 0.1

MAX_SEQ_LEN = 500

tokenizer_path = "models/e_nfa_to_dfa/e_nfa_to_dfa_tokenizer.pkl"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, ffn_hidden_dim, dropout_rate, max_len):
        super().__init__()
        self.src_embedding = nn.Embedding(vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        self.pos_decoder = PositionalEncoding(embed_size, max_len)

        self.transformer = nn.Transformer(
            d_model=embed_size, nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, tgt, pad_idx):
        src_pad_mask = (src == pad_idx)
        tgt_pad_mask = (tgt == pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)

        src_embedded = self.dropout(self.pos_encoder(self.src_embedding(src)))
        tgt_embedded = self.dropout(self.pos_decoder(self.tgt_embedding(tgt)))

        output = self.transformer(
            src_embedded, tgt_embedded,
            src_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.fc_out(output)

def load_e_nfa_to_dfa_model(model_path):
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer['stoi'])

    model = Seq2SeqTransformer(
        vocab_size, EMBED_SIZE, NUM_HEADS,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
        FFN_HIDDEN_DIM, DROPOUT_RATE, MAX_SEQ_LEN
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def tokenize_sequence(seq, tokenizer):
    tokens_list = [tokenizer['sos_token']] + list(seq) + [tokenizer['eos_token']]
    token_ids = [tokenizer['stoi'].get(ch, tokenizer['stoi'][tokenizer['pad_token']]) for ch in tokens_list]
    padded = token_ids[:MAX_SEQ_LEN] + [tokenizer['stoi'][tokenizer['pad_token']]] * (MAX_SEQ_LEN - len(token_ids))
    return padded

def predict_e_nfa_to_dfa(model, input_str):
    tokenizer = load_tokenizer(tokenizer_path)
    stoi, itos = tokenizer['stoi'], tokenizer['itos']
    pad_idx = stoi[tokenizer['pad_token']]

    input_ids = tokenize_sequence(input_str, tokenizer)
    src_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    output_ids = [stoi[tokenizer['sos_token']]]

    with torch.no_grad():
        for _ in range(MAX_SEQ_LEN):
            tgt_tensor = torch.tensor([output_ids], dtype=torch.long, device=DEVICE)
            logits = model(src_tensor, tgt_tensor, pad_idx)
            next_id = logits[0, -1].argmax().item()
            output_ids.append(next_id)

            if next_id == stoi[tokenizer['eos_token']]:
                break

    return ''.join(itos[idx] for idx in output_ids[1:-1])