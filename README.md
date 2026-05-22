# delta-mem tests

## References
* https://venturebeat.com/orchestration/a-0-12-parameter-add-on-gives-ai-agents-the-working-memory-rag-cant
* https://github.com/declare-lab/delta-Mem
* https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct

## Premise

We are going to review the article, code, weights and other details we can find to look to run this locally over a qwen3 model for comparison to their results.

Once we are sure that we can run it on one or other environment we'll want to refine how we are running if that it something awkward like python. We might want to offer patches to LlmSharp to run in C#, or TorchSharp example perhaps? We will also want to review applying the same technique over qwen3.6 and seeing it it maps with weights, how reliable it is, etc. If we can demo the same on a much newer model it is worth sharing that alone.