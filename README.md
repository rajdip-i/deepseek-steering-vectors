# DeepSeek Steering Vector Experimentation

An experimental project on activation steering with `deepseek-ai/deepseek-llm-7b-chat` using the Anthropic HH-RLHF dataset.

This project studies whether a simple steering vector, computed from preferred vs rejected responses, can shift a model's behavior at inference time toward more helpful and better-structured answers.

## Project Goal

The goal of this project is to compute a linear direction in the hidden-state space of a language model that represents a behavioral shift toward preferred responses, then inject that direction back into the model during generation.

In simple terms:

- take positive and negative response pairs
- measure how their hidden states differ
- average those differences into one normalized vector
- add that vector back into a chosen transformer layer during inference
- compare base vs steered outputs

## Why This Project

Large language models often contain interpretable behavioral directions in their internal representations. If those directions can be isolated, we may be able to influence model behavior without fine-tuning the full model.

This project was built to explore:

- whether helpfulness-related behavior can be approximated as a latent direction
- how activation steering behaves on a stronger open model like DeepSeek 7B
- whether lightweight runtime intervention can improve response quality
- what practical engineering issues appear when doing this in Kaggle GPU notebooks

## Final Setup

This repository is finalized around the DeepSeek-based experiment only.

Model:

- `deepseek-ai/deepseek-llm-7b-chat`

Dataset:

- `Anthropic/hh-rlhf`

Environment:

- Kaggle GPU notebook
- Hugging Face login via `HF_TOKEN`
- full GPU precision loading (`fp16` or `bf16` depending on support)


## Method

### 1. Load the chat model

We use the DeepSeek 7B chat checkpoint and run generation with chat formatting rather than plain raw prompts.

This was important because chat-tuned models behave much better when prompted through their chat template.

### 2. Load contrastive preference data

From `Anthropic/hh-rlhf`, we use:

- `chosen` as the positive example
- `rejected` as the negative example

These are already full conversation-style responses, so no separate prompt reconstruction is needed.

### 3. Extract hidden states

For each positive and negative sample:

- tokenize the text
- run the model with `output_hidden_states=True`
- select one transformer layer
- take the last-token hidden state

### 4. Compute the steering vector

We average hidden states across positive and negative examples separately:

`v = mean(hidden_positive) - mean(hidden_negative)`

Then normalize:

`v = v / ||v||`

This vector is interpreted as a latent direction associated with preferred responses.

### 5. Inject the vector during generation

During inference, a forward hook is attached to a selected transformer layer.

The hook adds:

`alpha * v`

to the hidden state of the final token position. Steering only the last token was more stable than modifying the full sequence.

### 6. Compare base vs steered outputs

We generate with and without the hook and compare:

- fluency
- helpfulness
- structure
- repetition or collapse behavior

## Key Engineering Decisions

Several practical issues showed up during experimentation, and the final notebook was shaped around fixing them.

### Correct dataset handling

An early mistake was assuming the HH dataset had a separate prompt field to combine with answers. In practice, `chosen` and `rejected` already contain the relevant conversation content.

### Chat formatting matters

Using a chat model as if it were a raw completion model gave weaker and less reliable outputs. Switching to chat-template-based prompting improved generation quality significantly.

### Safe hook registration

Notebook reruns can leave stale hooks attached to layers. The final notebook explicitly clears hooks before adding a new one.

### Device-safe steering

With automatic device mapping, different layers may live on different devices. The steering tensor must always be moved to the same device and dtype as the hooked hidden state.

### Mild steering works better

Large `alpha` values made outputs unstable or overly distorted. Smaller values were more usable.

## Main Findings

### 1. DeepSeek 7B was a much better fit than the smaller baseline

The final DeepSeek-based setup produced much more coherent outputs and showed a clearer behavioral shift under steering.

### 2. Steering produced visible but moderate changes

The steered outputs were often:

- slightly more explanatory
- better structured
- more aligned with helpful-answer style

The effect was noticeable but not extreme, which is often a good sign for stability.

### 3. Prompt formatting had a major effect

The quality difference from correct chat formatting was larger than many of the earlier steering tweaks. This was one of the most important lessons in the project.

### 4. Last-token steering was more stable

Applying the steering vector only to the final token position reduced collapse and odd generations compared with more aggressive interventions.

### 5. Decoding and generation settings matter

Some apparent model failures were actually decoding issues or generation artifacts:

- spacing/token-marker artifacts
- repetition tails
- stale generation config behavior

These had to be fixed before the research result was interpretable.

## Example Outcome

In final comparisons, the base and steered responses were both coherent, but the steered output often became slightly clearer and more instructional. This suggests that even a simple mean-difference activation vector can shift behavior in a useful direction on a 7B chat model.

## Limitations

- this is a small-scale inference-time experiment, not a full behavioral evaluation
- results depend heavily on layer choice and steering strength
- preference directions from HH-RLHF may mix helpfulness, harmlessness, and style
- subjective output comparison is useful, but not enough for rigorous benchmarking

## Future Work

- run a layer sweep across multiple candidate layers
- evaluate several `alpha` values systematically
- score outputs with automatic helpfulness and safety metrics
- compare last-token steering against full-sequence steering
- test covariance-aware or whitened steering vectors
- build a small benchmark set of prompts for consistent comparison


## Takeaway

This project shows that activation steering can be explored with a relatively simple pipeline:

- preference data
- hidden-state extraction
- vector construction
- runtime intervention

On DeepSeek 7B, the approach was capable of producing small but meaningful changes in response behavior, especially after the implementation details were made robust.
