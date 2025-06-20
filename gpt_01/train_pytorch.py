import torch
import torch.nn as nn
import torch.optim as optim
from model_pytorch import CharLSTM

if __name__ == "__main__":
    # Load data
    with open("gpt_01/examples/cursor_text_examples.txt") as f:
        lines = f.read().strip().split('\n')
    text = '\n'.join(lines)
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)

    # Prepare data (character-level)
    seq_len = 20
    data = []
    for line in lines:
        for i in range(len(line) - seq_len):
            seq = line[i:i+seq_len]
            target = line[i+1:i+seq_len+1]
            data.append((seq, target))
    if not data:
        print("Not enough data for training.")
        exit()

    X = torch.tensor([[stoi[c] for c in seq] for seq, _ in data], dtype=torch.long)
    Y = torch.tensor([[stoi[c] for c in tgt] for _, tgt in data], dtype=torch.long)

    # Model
    model = CharLSTM(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Generate samples
    print("\nSample generated text (PyTorch LSTM):")
    for _ in range(5):
        start = '\n'  # or random.choice(chars)
        print(model.generate(start, stoi, itos, max_len=40)) 