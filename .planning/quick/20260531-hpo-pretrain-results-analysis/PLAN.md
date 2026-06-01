---
slug: hpo-pretrain-results-analysis
description: Analyze aborted HPO pretrain run, extract best params, update config/pretrain.yaml
---

## Task

1. Analyze ClearML task f2f94e3c (hpo_pretrain, stopped) — extract trial results and best params
2. Update config/pretrain.yaml with best-trial hyperparameters
3. Report findings and suggest next steps

## Steps

- [x] Fetch HPO orchestrator task scalars (CER, LR, blank_frac)
- [x] Download checkpoint_pretrain artifact and inspect model architecture
- [x] Identify trial boundaries and best trial from scalar data
- [x] Extract hyperparams (rnn_hidden, num_layers, pretrain_lr, pretrain_epochs, batch_size)
- [x] Update config/pretrain.yaml
- [x] Commit
