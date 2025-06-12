import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

# Define the vocabulary
INPUT_SYMBOLS = ['a', 'b', 'c', 'd', 'ε', '&']
STACK_SYMBOLS = ['Z', 'A', 'B', 'C', 'D']
STATES = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'qf']
ACTIONS = ['PUSH', 'POP', 'NOOP']

# Define special padding token
PAD_TOKEN = '<PAD>'

#define max input string length and maximum number of transitions
MAX_INPUT_LEN = 125
MODEL_CONFIG = {
            'input_vocab_size': len(INPUT_SYMBOLS) + 1,
            'state_vocab_size': len(STATES) + 1,
            'stack_vocab_size': len(STACK_SYMBOLS) + 1,
            'action_vocab_size': len(ACTIONS) + 1,
            'd_model': 256,
            'nhead': 4,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'max_transitions': 123,        # same as training
            'pos_enc_max_len': 256         # <<< this matches the checkpoint
        }


# Create mappings for tokenization with explicit padding token
def create_vocabulary_mappings():
    """Create token-to-index and index-to-token mappings for all vocabularies."""
    # Add padding token to all vocabularies
    input_symbols_with_pad = [PAD_TOKEN] + INPUT_SYMBOLS
    stack_symbols_with_pad = [PAD_TOKEN] + STACK_SYMBOLS
    states_with_pad = [PAD_TOKEN] + STATES
    actions_with_pad = [PAD_TOKEN] + ACTIONS

    # Create token to index mappings
    input_symbol_to_idx = {symbol: idx for idx, symbol in enumerate(input_symbols_with_pad)}
    stack_symbol_to_idx = {symbol: idx for idx, symbol in enumerate(stack_symbols_with_pad)}
    state_to_idx = {state: idx for idx, state in enumerate(states_with_pad)}
    action_to_idx = {action: idx for idx, action in enumerate(actions_with_pad)}

    # Create index to token mappings
    idx_to_input_symbol = {idx: symbol for symbol, idx in input_symbol_to_idx.items()}
    idx_to_stack_symbol = {idx: symbol for symbol, idx in stack_symbol_to_idx.items()}
    idx_to_state = {idx: state for state, idx in state_to_idx.items()}
    idx_to_action = {idx: action for action, idx in action_to_idx.items()}

    return (input_symbol_to_idx, stack_symbol_to_idx, state_to_idx, action_to_idx,
            idx_to_input_symbol, idx_to_stack_symbol, idx_to_state, idx_to_action)

# Unpack the mappings
(input_symbol_to_idx, stack_symbol_to_idx, state_to_idx, action_to_idx,
 idx_to_input_symbol, idx_to_stack_symbol, idx_to_state, idx_to_action) = create_vocabulary_mappings()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_INPUT_LEN + 10, dropout: float = 0.1):
        """
        Initialize the positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum length of the sequences
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class TransformerPDA(nn.Module):
    def __init__(self, input_vocab_size: int, state_vocab_size: int, stack_vocab_size: int,
             action_vocab_size: int, d_model: int = 128, nhead: int = 8,
             num_encoder_layers: int = 6, num_decoder_layers: int = 6,
             dim_feedforward: int = 512, dropout: float = 0.1, *,
             max_transitions: int = 2 * (MAX_INPUT_LEN - 1) + 3,
             pos_enc_max_len: int = MAX_INPUT_LEN + 10):
        """
        Initialize the Transformer PDA model.

        Args:
            input_vocab_size: Size of the input vocabulary
            state_vocab_size: Size of the state vocabulary
            stack_vocab_size: Size of the stack vocabulary
            action_vocab_size: Size of the action vocabulary
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            max_transitions: Maximum number of transitions
        """
        super(TransformerPDA, self).__init__()

        self.max_transitions = max_transitions
        self.d_model = d_model
        self.teacher_forcing_ratio = 0.5

        # Embeddings with explicit padding_idx=0
        self.input_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0)
        self.from_state_embedding = nn.Embedding(state_vocab_size, d_model, padding_idx=0)
        self.input_symbol_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0)
        self.stack_symbol_embedding = nn.Embedding(stack_vocab_size, d_model, padding_idx=0)
        self.action_embedding = nn.Embedding(action_vocab_size, d_model, padding_idx=0)

        # Positional encoding with dropout
        self.positional_encoding = PositionalEncoding(d_model, max_len=pos_enc_max_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Output layers for each component of the transition
        self.input_symbol_fc = nn.Linear(d_model, input_vocab_size)
        self.stack_symbol_fc = nn.Linear(d_model, stack_vocab_size)
        self.to_state_fc = nn.Linear(d_model, state_vocab_size)
        self.action_fc = nn.Linear(d_model, action_vocab_size)


    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence.

        Args:
            sz: Sequence length

        Returns:
            Mask tensor of shape (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _embed_step(self, step_tensor: torch.Tensor) -> torch.Tensor:
        """
        step_tensor shape: (batch, 4)
        columns: [from_state , input_symbol , stack_symbol , prev_action]
        """
        fs, inp, stk, act = step_tensor.T
        return ( self.from_state_embedding(fs) +
                self.input_symbol_embedding(inp) +
                self.stack_symbol_embedding(stk) +
                self.action_embedding(act) )

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        src : (batch, src_len)
        tgt : (batch, tgt_len, 5)  –– full gold sequence incl. final ε-transition
        """

        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        batch, device = src.size(0), src.device
        src_pad_mask = (src == 0)

        # ---------- Encoder ----------
        sqrt_d = self.d_model ** 0.5 # keeps the value on the Torch graph
        src_emb = self.positional_encoding(self.input_embedding(src) * sqrt_d)
        memory  = self.transformer_encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # ---------- Inference mode ----------
        if tgt is None:
            return self._generate_transitions(memory, batch, device, src)

        # ---------- Teacher-forced decoding ----------
        tgt_len = tgt.size(1)
        logits_per_t = []           # collect logits for loss
        # --- keep train/infer identical ---------------------------------
        shadow_stacks = [['Z'] for _ in range(batch)]
        prev_action   = torch.full((batch,), action_to_idx['NOOP'], device=device)
        step_input = torch.stack([
            torch.full((batch,), state_to_idx['q0'], device=device),        # from_state
            torch.full((batch,), input_symbol_to_idx['ε'], device=device),  # current input symbol
            torch.full((batch,), stack_symbol_to_idx['Z'], device=device),  # stack top
            prev_action                                                     # action executed before t=0
        ], dim=1)

        for t in range(tgt_len):
            # embed current step (1 token length for decoder)
            step_emb = self._embed_step(step_input).unsqueeze(1)
            sqrt_d = self.d_model ** 0.5
            step_emb = self.positional_encoding(step_emb * sqrt_d)

            dec_out = self.transformer_decoder(step_emb, memory)[:, 0]   # (batch, d_model)

            # project to vocabularies
            is_log = self.input_symbol_fc(dec_out)
            ss_log = self.stack_symbol_fc(dec_out)
            ts_log = self.to_state_fc(dec_out)
            ac_log = self.action_fc(dec_out)

            logits_per_t.append((is_log, ss_log, ts_log, ac_log))

            # scheduled sampling
            # ---------- scheduled sampling & stack simulation ----------
            use_gold  = (torch.rand(batch, device=device) < teacher_forcing_ratio)
            next_step = torch.empty_like(step_input)

            # (1) gold branch
            if use_gold.any():
                g = use_gold
                next_step[g, 0] = tgt[g, t, 3]          # to_state_t  becomes from_state_{t+1}
                # --- safe look-ahead -------------------------------------------------
                if t < tgt_len - 1:                     # still have a gold step ahead
                    next_step[g, 1] = tgt[g, t + 1, 1]  # next input symbol
                    next_step[g, 2] = tgt[g, t + 1, 2]  # next stack top
                else:                                   # this is the final gold transition
                    next_step[g, 1] = input_symbol_to_idx['ε']   # dummy ε
                    next_step[g, 2] = stack_symbol_to_idx['Z']   # dummy Z
                next_step[g, 3] = tgt[g, t, 4]          # prev_action_{t+1}

            # (2) sampled branch
            if (~use_gold).any():
                s = ~use_gold
                next_step[s, 0] = ts_log.argmax(-1)[s]
                next_step[s, 1] = is_log.argmax(-1)[s]
                next_step[s, 2] = ss_log.argmax(-1)[s]
                next_step[s, 3] = ac_log.argmax(-1)[s]

            # (3) execute prev_action on the shadow stack so stack_top is correct
            for b in range(batch):
                act_id = step_input[b, 3].item()
                if act_id == action_to_idx['PUSH']:
                    # push the UPPER-CASE form of the CURRENT INPUT SYMBOL
                    inp_idx = step_input[b, 1].item()          # column 1 = input symbol
                    inp_sym = idx_to_input_symbol[inp_idx]
                    if inp_sym not in {'ε', '&'}:
                        shadow_stacks[b].append(inp_sym.upper())
                elif act_id == action_to_idx['POP'] and shadow_stacks[b]:
                    shadow_stacks[b].pop()

                top_sym = shadow_stacks[b][-1] if shadow_stacks[b] else 'Z'
                next_step[b, 2] = stack_symbol_to_idx[top_sym]

            step_input = next_step


        # stack over time → (batch, tgt_len, ·)
        is_logits = torch.stack([l[0] for l in logits_per_t], dim=1)
        ss_logits = torch.stack([l[1] for l in logits_per_t], dim=1)
        ts_logits = torch.stack([l[2] for l in logits_per_t], dim=1)
        ac_logits = torch.stack([l[3] for l in logits_per_t], dim=1)

        # shift target to align with predictions (like seq-2-seq)
        tgt_output = tgt.clone()

        return is_logits, ss_logits, ts_logits, ac_logits, tgt_output


    def _generate_transitions(
        self,
        memory: torch.Tensor,
        batch_size: int,
        device: torch.device,
        input_tensor: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        """
        Autoregressively generate PDA transitions.
        The first element in every output column now corresponds to the first
        *real* transition (q0 → …); no placeholder rows are produced.
        """

        # ------ initial “cursor” values (state, input, stack, action) -------
        from_state   = torch.full((batch_size, 1), state_to_idx['q0'], device=device)
        input_symbol = torch.full((batch_size, 1), input_symbol_to_idx['ε'], device=device)
        stack_symbol = torch.full((batch_size, 1), stack_symbol_to_idx['Z'], device=device)
        prev_action  = torch.full((batch_size, 1), action_to_idx['NOOP'], device=device)

        # if we really do have an input string, copy its first symbol
        if input_tensor is not None and input_tensor.size(1) > 0:
            first_tok = input_tensor[:, 0].unsqueeze(1)          # (batch, 1)
            input_symbol = torch.where(first_tok == 0,
                                       input_symbol,             # keep ε where PAD
                                       first_tok)

        # ------ containers; **no** dummy row any more ­-----------------------
        all_from_states   : list[torch.Tensor] = []
        all_input_symbols : list[torch.Tensor] = []
        all_stack_symbols : list[torch.Tensor] = []
        all_to_states     : list[torch.Tensor] = []
        all_actions       : list[torch.Tensor] = []

        # book-keeping helpers
        input_positions     = torch.zeros(batch_size, dtype=torch.long, device=device)
        stacks              = [['Z'] for _ in range(batch_size)]
        reached_epsilon     = torch.zeros(batch_size, dtype=torch.bool, device=device)
        final_state_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # --------------------------------------------------------------------
        for _ in range(self.max_transitions):
            # --- update stack‐symbol column ---------------------------------
            for b in range(batch_size):
                top = stacks[b][-1] if stacks[b] else 'Z'
                stack_symbol[b, 0] = stack_symbol_to_idx[top]

            # --- embed current step -----------------------------------------
            # add *prev_action* so test-time matches training
            step_emb = (
                self.from_state_embedding(from_state.squeeze(1)) +
                self.input_symbol_embedding(input_symbol.squeeze(1)) +
                self.stack_symbol_embedding(stack_symbol.squeeze(1)) +
                self.action_embedding(prev_action.squeeze(1))     # NEW
            ).unsqueeze(1)                                    # (batch, 1, d_model)
            step_emb = self.positional_encoding(step_emb * (self.d_model ** 0.5))

            # --- run decoder / project to vocabularies ----------------------
            dec_out   = self.transformer_decoder(step_emb, memory)[:, 0]  # (batch,d)
            is_logit  = self.input_symbol_fc(dec_out)
            ss_logit  = self.stack_symbol_fc(dec_out)
            ts_logit  = self.to_state_fc(dec_out)
            ac_logit  = self.action_fc(dec_out)

            is_pred = is_logit.argmax(dim=-1, keepdim=True)   # (batch,1)
            ss_pred = ss_logit.argmax(dim=-1, keepdim=True)
            ts_pred = ts_logit.argmax(dim=-1, keepdim=True)
            ac_pred = ac_logit.argmax(dim=-1, keepdim=True)

            # --- store *real* transition ------------------------------------
            all_from_states.append(from_state.clone())
            all_input_symbols.append(input_symbol.clone())
            all_stack_symbols.append(stack_symbol.clone())
            all_to_states.append(ts_pred)
            all_actions.append(ac_pred)

            # --- update “cursor” for next turn ------------------------------
            from_state = ts_pred.clone()
            prev_action = ac_pred.clone()

            # shadow-stack simulation (needed for next stack top)
            for b in range(batch_size):
                act_id = ac_pred[b].item()
                if act_id == action_to_idx['PUSH']:
                    # ss_idx = ss_pred[b, 0].item()
                    inp_sym = idx_to_input_symbol[input_symbol[b, 0].item()]
                    if inp_sym not in {'ε', '&'}:                       # guard
                        stacks[b].append(inp_sym.upper())
                    #print(f"pushed symbol {idx_to_stack_symbol[ss_idx]}")
                elif act_id == action_to_idx['POP'] and len(stacks[b]) > 0:
                    stacks[b].pop()

            # advance through the input tape
            for b in range(batch_size):
                if final_state_reached[b]:
                    input_symbol[b, 0] = input_symbol_to_idx['ε']
                    continue

                if not reached_epsilon[b]:
                    pos = input_positions[b].item()
                    if input_tensor is not None and pos < input_tensor.size(1) - 1:
                        input_positions[b] += 1
                        nxt = input_tensor[b, pos + 1].item()
                        input_symbol[b, 0] = nxt if nxt != 0 else input_symbol_to_idx['ε']
                    else:
                        reached_epsilon[b] = True
                        input_symbol[b, 0] = input_symbol_to_idx['ε']

                # mark final state
                if idx_to_state[ts_pred[b].item()] == 'qf':
                    final_state_reached[b] = True

            if final_state_reached.all():
                break

        # ------ stack lists → tensors ---------------------------------------
        all_from_states   = torch.cat(all_from_states,   dim=1)
        all_input_symbols = torch.cat(all_input_symbols, dim=1)
        all_stack_symbols = torch.cat(all_stack_symbols, dim=1)
        all_to_states     = torch.cat(all_to_states,     dim=1)
        all_actions       = torch.cat(all_actions,       dim=1)

        return (all_from_states,
                all_input_symbols,
                all_stack_symbols,
                all_to_states,
                all_actions)
    
# Function to predict PDA transitions for an input string
def predict_PDA_transitions(model: nn.Module, input_str: str, device: str = 'cpu') -> List[str]:
    """
    Predict PDA transitions for an input string with improved filtering.

    Args:
        model: The trained model
        input_str: Input string
        device: Device to use for prediction

    Returns:
        List of predicted transition strings
    """
    # Add epsilon if not already present
    if not input_str.endswith('ε'):
        input_str = input_str + 'ε'

    # Tokenize the input
    input_tokens = [input_symbol_to_idx.get(c, 0) for c in input_str]
    # Pad to max length
    # input_tokens = input_tokens + [0] * (20 - len(input_tokens))
    pad = MAX_INPUT_LEN - len(input_tokens)
    if pad < 0:
        raise ValueError(f"Input too long (>{MAX_INPUT_LEN})")
    input_tokens += [0] * pad
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
        from_states, input_symbols, stack_symbols, to_states, actions = predictions

    # Convert to human-readable transitions
    transitions = []

    for i in range(min(len(from_states[0]), len(to_states[0]))):
        from_state_idx = from_states[0, i].item()
        input_symbol_idx = input_symbols[0, i].item()
        stack_symbol_idx = stack_symbols[0, i].item()
        to_state_idx = to_states[0, i].item()
        action_idx = actions[0, i].item()

        from_state = idx_to_state.get(from_state_idx, "<UNK>")
        input_symbol = idx_to_input_symbol.get(input_symbol_idx, "<UNK>")
        stack_symbol = idx_to_stack_symbol.get(stack_symbol_idx, "<UNK>")
        to_state = idx_to_state.get(to_state_idx, "<UNK>")
        action = idx_to_action.get(action_idx, "<UNK>")

        # Skip padding or unknown tokens
        if PAD_TOKEN in [from_state, input_symbol, stack_symbol, to_state, action] or \
           "<UNK>" in [from_state, input_symbol, stack_symbol, to_state, action]:
            continue

        # Skip invalid transitions (heuristic)
        if from_state == to_state and action == 'NOOP' and i and input_symbol == 'ε' > 0:
            continue

        transition = f"delta({from_state}, {input_symbol}, {stack_symbol}) -> ({to_state}, {action})"
        transitions.append(transition)

    # Filter out duplicates while preserving order
    seen = set()
    valid_transitions = []

    for t in transitions:
        if t in seen:
            continue

        valid_transitions.append(t)
        seen.add(t)

    return valid_transitions

# Function to load a trained model
def load_PDA_model(model_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load a trained model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint
        model_config: Model configuration dictionary
        device: Device to load the model on

    Returns:
        Loaded model
    """
    global MODEL_CONFIG
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        # ========= BEGIN PATCH (merge hparams) =========
        if MODEL_CONFIG is None:
            MODEL_CONFIG = {}
        MODEL_CONFIG.update(checkpoint.get("hparams", {}))

        # Create a new model if configuration is provided
        if MODEL_CONFIG is not None:
            model = TransformerPDA(**MODEL_CONFIG)
        else:
            # Default configuration if none provided
            model = TransformerPDA(
                input_vocab_size=len(INPUT_SYMBOLS) + 1,  # +1 for padding
                state_vocab_size=len(STATES) + 1,
                stack_vocab_size=len(STACK_SYMBOLS) + 1,
                action_vocab_size=len(ACTIONS) + 1,
                d_model=64,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=128,
                dropout=0.1,          # Slightly higher dropout for better generalization
                max_transitions=2*(MAX_INPUT_LEN - 1) + 3,              # 2*60 + 3
                pos_enc_max_len=MAX_INPUT_LEN + 10,              # see sect. 1-b
            )

        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}, Val Loss: {checkpoint.get('val_loss', 'unknown')}")

        return model

    except FileNotFoundError:
        print(f"Model file {model_path} not found")
        raise FileNotFoundError(f"Model file {model_path} not found")

    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        raise e