import torch
import torch.nn as nn
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    num_epochs,
    grad_clip,
    device=device,
):
    epoch_losses = []

    for ep in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_in)  # (B, T, V)
            B, T, V = logits.size()

            loss = criterion(
                logits.reshape(B * T, V),
                tgt_out.reshape(B * T),
            )
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {ep:03d} | Loss {avg_loss:.4f}")

    return epoch_losses



def build_model_1a():
    return Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        emb_dim=EMB_DIM_1,
        num_layers=N_LAYERS_1,
        num_heads=N_HEADS_1,
        forward_dim=FF_DIM_1,
        dropout=DROPOUT_1,
        max_len=MAX_LEN_1
    )

