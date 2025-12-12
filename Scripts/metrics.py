def _strip_sequence(seq, pad_idx, eos_idx):
    """Remove PAD, cut at EOS"""
    out = []
    for idx in seq:
        if idx == eos_idx:
            break
        if idx == pad_idx:
            continue
        out.append(idx)
    return out

def sequence_accuracy_strict(pred, gold, pad_idx, eos_idx):
    p = _strip_sequence(pred, pad_idx, eos_idx)
    g = _strip_sequence(gold, pad_idx, eos_idx)
    if p != g:
        return 0.0
    # require exactly one EOS and no junk after
    if eos_idx in pred:
        eos_pos = pred.index(eos_idx)
        if any(tok != pad_idx for tok in pred[eos_pos+1:]):
            return 0.0

    return 1.0

def token_accuracy(pred_indices, gold_indices, pad_idx, eos_idx):
    # Clean gold
    gold = []
    for t in gold_indices:
        if t == eos_idx:
            break
        if t != pad_idx:
            gold.append(t)

    correct = 0
    total = len(gold)

    # Compare only positions in gold
    for i, g in enumerate(gold):
        if i < len(pred_indices) and pred_indices[i] == g:
            correct += 1

    # If prediction shorter: missing tokens count as incorrect implicitly
    return correct, total

def token_accuracy_exp1(pred_indices, gold_indices, pad_idx, eos_idx):

    p = _strip_sequence(pred_indices, pad_idx, eos_idx)
    g = _strip_sequence(gold_indices, pad_idx, eos_idx)

    min_len = min(len(p), len(g))
    correct = sum(p[i] == g[i] for i in range(min_len))

    return correct, min_len