# ADED: Adaptive Draft-Verification for Efficient Large Language Model Decoding


## Introduction

Large language model (LLM) decoding involves generating a sequence of tokens based on a given context, where each token is predicted one at a time using the model's learned probabilities. The typical autoregressive decoding method requires a separate forward pass through the model for each token generated, which is computationally inefficient and poses challenges for deploying LLMs in latency-sensitive scenarios. The main limitations of current decoding methods stem from their inefficiencies and resource demands. Existing approaches either necessitate fine-tuning smaller models, which is resource-intensive, or rely on fixed retrieval schemes to construct drafts for the next tokens, which lack adaptability and fail to generalize across different models and contexts. To address these issues, we introduce a novel methodology called ADED, which accelerates LLM decoding without requiring fine-tuning. Our approach involves an adaptive draft-verification process that evolves over time to improve efficiency. We utilize a tri-gram matrix-based LLM representation to dynamically approximate the output distribution of the LLM, allowing the model to adjust to changing token probabilities during the decoding process. Additionally, we implement a draft construction mechanism that effectively balances exploration and exploitation, ensuring that the drafts generated are both diverse and close to the true output distribution of the LLM. The importance of this design lies in its ability to optimize the draft distribution adaptively, leading to faster and more accurate decoding. Through extensive experiments on various benchmark datasets and LLM architectures, we demonstrate that ADED significantly accelerates the decoding process while maintaining high accuracy, making it suitable for deployment in a wide range of practical applications.

## Installation
```bash
pip install -r requirements.txt
```
Then follow the `README.md` in DraftRetriever.

## Build Corpus

Build a chat datastore using data from [UltraChat](https://huggingface.co/datasets/stingning/ultrachat) 
```bash
cd datastore
python3 get_datastore_chat.py --model-path lmsys/vicuna-7b-v1.5 --large-datastore True 
```
Build a Python code generation datastore from [The Stack](https://huggingface.co/datasets/bigcode/the-stack) 
```bash
cd datastore
python3 get_datastore_code.py --model-path codellama/CodeLlama-7b-instruct-hf --large-datastore True 
```

The corpus generated using the above commands will be larger than the data presented in our paper. This is because, for the convenience of testing the impact of corpus pruning, the generated corpus here retains the complete 3-gram (unpruned). The corpus will be pruned during reading, keeping only the top-12 entries with the highest probabilities (changeable as needed).

## Inference

### Inference on MT-Bench
```bash
cd llm_judge
CUDA_VISIBLE_DEVICES=0 python get_model_answer_aded.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --datastore-path ../datastore/datastore_chat_large.idx
```

### Inference on HumanEval
```bash
cd human_eval
CUDA_VISIBLE_DEVICES=0 python aded_test.py --model-path lmsys/vicuna-7b-v1.5 --datastore-path ../datastore/datastore_stack_large.idx
```


## Acknowledgements
The codebase is mainly from [REST](https://github.com/FasterDecoding/REST), some code is from [Medusa](https://github.com/FasterDecoding/Medusa) and influenced by remarkable projects from the LLM community, including [FastChat](https://github.com/lm-sys/FastChat), [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/), [vllm](https://github.com/vllm-project/vllm) and many others.

