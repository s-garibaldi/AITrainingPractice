import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLSTM(nn.Module):
    """A simple character-level LSTM language model for text generation."""
    def __init__(self, vocab_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """Forward pass through the model."""
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden) if hidden is not None else self.lstm(x)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_seq, stoi, itos, max_len=40):
        """Generate text starting from start_seq up to max_len characters."""
        self.eval()
        input_seq = torch.tensor([[stoi[c] for c in start_seq]], dtype=torch.long)
        hidden = None
        out_seq = list(start_seq)
        for _ in range(max_len - len(start_seq)):
            logits, hidden = self.forward(input_seq, hidden)
            next_char_logits = logits[0, -1]
            probs = F.softmax(next_char_logits, dim=0).detach().cpu().numpy()
            next_idx = torch.multinomial(torch.tensor(probs), 1).item()
            next_char = itos[next_idx]
            out_seq.append(next_char)
            input_seq = torch.tensor([[next_idx]], dtype=torch.long)
        return ''.join(out_seq) 