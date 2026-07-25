import torch

def apply_sequence_matching(similarity_matrix, seq_length, eps=1e-8):
    """Sequence matching per Joseph, Fischer & Milford (arXiv:2509.01968, Sec. III-D).

    For each query q:
      1. Causal history submatrix C = S[q-N+1 : q+1, :], N = min(L, q+1)
         (dynamic history length at the top boundary).
      2. Dual-axis z-score normalisation of the RAW similarity submatrix C
         (per-column over the N history frames, then per-row over the R
         references). NOTE: the paper says "across columns and then across
         rows" without releasing code yet; this is the most natural reading.
         Swap the two blocks if their released code differs.
      3. Score for reference j = trace of the most recent k x k diagonal
         block ending at (q, j), k = min(N, j+1) (adaptive at the left
         reference boundary).

    Runs on whatever device `similarity_matrix` is on. Returns a new tensor;
    does not modify the input.
    """
    S = similarity_matrix
    device = S.device
    Q, R = S.shape
    out = torch.empty_like(S)

    for q in range(Q):
        N = min(seq_length, q + 1)
        C = S[q - N + 1: q + 1, :].clone()  # (N, R)

        # dual-axis z-score on the raw similarity submatrix 
        if N > 1:  # column stats undefined for a single history frame
            C = (C - C.mean(dim=0, keepdim=True)) / (
                C.std(dim=0, keepdim=True, unbiased=False) + eps)
        C = (C - C.mean(dim=1, keepdim=True)) / (
            C.std(dim=1, keepdim=True, unbiased=False) + eps)

        scores = torch.empty(R, device=device, dtype=S.dtype)

        # main region (j >= N-1): full-length diagonal trace, k = N
        if R >= N:
            rows = torch.arange(N, device=device).unsqueeze(1)              # (N, 1)
            cols = (torch.arange(N - 1, R, device=device).unsqueeze(0)
                    + (rows - (N - 1)))                                     # (N, R-N+1)
            scores[N - 1:] = C[rows.expand_as(cols), cols].sum(dim=0)

        # left boundary (j < N-1): shrink kernel, k = j+1
        for j in range(min(N - 1, R)):
            k = j + 1
            i = torch.arange(k, device=device)
            scores[j] = C[N - k + i, i].sum()

        out[q] = scores

    return out