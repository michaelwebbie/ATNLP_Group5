
import torch
from Scripts.decoder import greedy_decode
from Scripts.metrics import token_accuracy, sequence_accuracy, token_accuracy_exp1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_exp1_token(model, data_loader, tgt_vocab, max_len=50, device=device):
    sos_idx = tgt_vocab.stoi["<SOS>"]
    eos_idx = tgt_vocab.stoi["<EOS>"]
    pad_idx = tgt_vocab.stoi["<PAD>"]

    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for src, _, tgt_out in data_loader:
            src = src.to(device)
            tgt_out = tgt_out.to(device)

            preds_batch = greedy_decode(
                model,
                src,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=max_len,
                device=device,
            )

            for i in range(src.size(0)):
                gold = tgt_out[i].tolist()
                # Let metric handle PAD/EOS via pad_idx/eos_idx
                pred = preds_batch[i]
                c, t = token_accuracy_exp1(pred, gold, pad_idx, eos_idx)
                total_correct += c
                total_tokens += t

    return total_correct / total_tokens if total_tokens > 0 else 0.0

def evaluate_exp2(
    model,
    loader,
    sos_idx,
    eos_idx,
    pad_idx,
    device,
    use_oracle=False,
):
    """
    Returns: (sequence_accuracy %, token_accuracy %)
    """

    total_seq = 0
    total_seq_correct = 0

    total_tok_correct = 0
    total_tok_total = 0

    model.eval()

    with torch.no_grad():
        for batch_src, _, batch_tgt_out in loader:
            batch_src     = batch_src.to(device)
            batch_tgt_out = batch_tgt_out.to(device)

            B = batch_src.size(0)

            for i in range(B):
                src_i  = batch_src[i].unsqueeze(0)
                gold_i = batch_tgt_out[i].tolist()

                # Clean gold
                gold_clean = []
                for t in gold_i:
                    if t == eos_idx:
                        break
                    if t != pad_idx:
                        gold_clean.append(t)

                gold_len = len(gold_clean)

                # Predict
                if use_oracle:
                    pred = greedy_decode(
                        model,
                        src_i,
                        sos_idx=sos_idx,
                        eos_idx=eos_idx,
                        max_len=gold_len,
                        oracle_length=gold_len,
                        forbid_eos=True,
                        device=device,
                    )[0]
                else:
                    pred = greedy_decode(
                        model,
                        src_i,
                        sos_idx=sos_idx,
                        eos_idx=eos_idx,
                        max_len=60,
                        device=device,
                    )[0]

                # token-level accuracy
                c_tok, t_tok = token_accuracy(
                    pred,
                    gold_i,
                    pad_idx=pad_idx,
                    eos_idx=eos_idx,
                )

                total_tok_correct += c_tok
                total_tok_total   += t_tok

                # sequence accuracy
                seq_ok = sequence_accuracy(
                    pred,
                    gold_i,
                    pad_idx=pad_idx,
                    eos_idx=eos_idx,
                )
                total_seq_correct += seq_ok
                total_seq += 1

    tok_acc = 100 * total_tok_correct / total_tok_total
    seq_acc = 100 * total_seq_correct / total_seq

    return seq_acc, tok_acc

def evaluate_exp3(model, data_loader, tgt_vocab, max_len, device):
    model.eval()
    pad_idx = tgt_vocab.stoi["<PAD>"]

    total_seq = 0
    total_tok_correct = 0
    total_tok_total = 0

    with torch.no_grad():
        for src, _, tgt_out in data_loader:
            src = src.to(device)
            tgt_out = tgt_out.to(device)

            preds = greedy_decode(
                model,
                src,
                sos_idx=tgt_vocab.stoi["<SOS>"],
                eos_idx=tgt_vocab.stoi["<EOS>"],
                max_len=max_len,
                device=device,
            )

            for pred, gold in zip(preds, tgt_out):
                gold = gold.tolist()

                # token acc
                c, t = token_accuracy(
                    pred,
                    gold,
                    pad_idx=pad_idx,
                    eos_idx=tgt_vocab.stoi["<EOS>"],
                )
                total_tok_correct += c
                total_tok_total += t

                # seq acc
                if sequence_accuracy(
                    pred,
                    gold,
                    pad_idx=pad_idx,
                    eos_idx=tgt_vocab.stoi["<EOS>"],
                ):
                    total_seq += 1

    seq_acc = 100 * total_seq / len(data_loader.dataset)
    tok_acc = 100 * total_tok_correct / total_tok_total

    return seq_acc, tok_acc

