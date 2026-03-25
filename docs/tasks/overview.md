# Tasks Overview

STEB evaluates style text embeddings across six task types. Each task probes a different aspect of how well embeddings capture stylistic properties.

| Task | What It Measures | Primary Metric |
|---|---|---|
| [Clustering](clustering.md) | Style-based cluster formation | V-measure |
| [All-to-All Pair Classification](all-to-all-pair-classification.md) | Same-vs-different style discrimination | EER, AUC |
| [Pre-defined Pair Classification](pre-defined-pair-classification.md) | Verification on pre-defined pairs | EER, AUC |
| [Order Alignment](order-alignment.md) | Preservation of graded style ordering | Accuracy |
| [Retrieval](retrieval.md) | Style-matched text retrieval | MRR, Recall@K |
| [Probing](probing.md) | Linguistic property encoding | Accuracy |
