## GPU inference lab

### Goal

Build an end-to-end lab for training, packaging, deploying, and optimizing a small Qwen-based product reranker on Azure ML.

Use the Amazon ESCI dataset, filtered to English. ESCI provides a query, a set of candidate products, and graded relevance labels (`E > C > S > I`). Retrieval is out of scope initially: treat the candidate set supplied by ESCI as the output of an upstream retrieval stage.

The initial model should score each `(query, product)` pair independently. Scores are then used to rank all candidates for a query and evaluate ranking quality, primarily with NDCG.

### Flow

1. ~~Set up AML workspace and T4 GPU compute~~
2. ~~Run a test GPU job~~
3. Prepare project bootstrap/data scripts:

   * ~~Prepare/filter the local ESCI snapshot to English.~~
   * ~~Use a pinned local Qwen3-Reranker-0.6B checkpoint.~~
   * ~~Create an idempotent uploader for versioned AML data/model assets.~~
   * Run the bootstrap and verify the registered assets in the workspace.
4. Fine-tune Qwen for product relevance scoring.
5. Evaluate model quality on held-out ESCI queries using ranking metrics such as NDCG.
6. Package and deploy the trained model using a modern GPU inference runtime.
7. Build a C++ traffic generator that simulates production search traffic.
8. Benchmark and optimize inference, including:

   * batching candidates from one query;
   * continuous batching across queries;
   * shared query-prefix / KV-cache reuse;
   * different candidate counts and input lengths;
   * concurrency, throughput, latency, GPU utilization, and memory usage;
   * later, quantization and alternative runtimes.

### Main learning objective

Use this project to understand the complete transformer inference lifecycle:

`dataset -> fine-tuning -> evaluation -> model artifact -> GPU serving -> load testing -> inference optimization`

The main focus is not maximizing ESCI ranking quality, but gaining hands-on understanding of transformer GPU inference, prefill/KV-cache behavior, batching, capacity, and serving trade-offs.
