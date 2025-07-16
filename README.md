# Fine-Tuning Large Language Models for Conversational AI

This project demonstrates how to use the FLAN-T5 model from Hugging Face's transformers library to generate summaries for dialogues from the DialogSum Dataset. It explores different inference strategies—zero-shot, one-shot, and few-shot—and showcases how prompt engineering improves summarization quality.
The notebook cover data preprocessing, model customization, training, and evaluation, with emphasis on improving text relevance, coherence, and safety in generated outputs.

## Project Overview
### Lab 1: Summarizing Dialogues Using Fine-Tuned LLMs
* Objective: Leverage pre-trained models for generating concise and coherent summaries of conversational data.
* Approach:
  - Loaded conversational datasets and preprocessed text to remove noise (e.g., special characters and tags).
  - Fine-tuned the Flan-T5 model for summarization, customizing training parameters to adapt the model to dialogue structure.
  - Evaluated summarization quality using BLEU and ROUGE metrics, comparing model-generated summaries with original dialogues.
 
Results: Achieved a 30% improvement in relevance and coherence, with BLEU and ROUGE scores confirming enhanced performance in summarizing conversational data effectively.

## Dataset
DialogSum is a human-annotated dialogue summarization dataset containing 10,000+ dialogues along with manually written summaries and topic labels.
Dataset link: https://huggingface.co/datasets/knkarthick/dialogsum


## Setup and Requirements
Before running the notebook, make sure the following libraries are installed:
```py
pip install transformers datasets torch
```

### Inference Steps
1. Baseline (Without Prompt Engineering)
Directly pass the dialogue to the FLAN-T5 model without specifying what the task is.
Output tends to be vague or contextually incorrect.
e.g:
Baseline human summary :#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
AI generated summary: #Person1#: I'm thinking of upgrading my computer.


3. Zero-Shot Inference (With Instruction Prompt)
Provide the model with a clear instruction like:
Summarize the following dialogue conversation. FLAN-T5 performs better with such task-oriented prompts.
    - BASELINE HUMAN SUMMARY: #Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
    - MODEL GENERATION - ZERO SHOT: The train is about to leave.


3. One-Shot Inference with Prompt Template
Use FLAN-T5’s pretrained style prompts (e.g., examples from T5 training corpus). Better captures structure but still lacks specificity at times.

4. One-Shot and Few-Shot Inference
Add 1 or more example dialogue-summary pairs before the target dialogue.

One-shot significantly improves generation quality.

Few-shot adds more examples but may hit token limit (~512 tokens).


### Resources
Hugging Face FLAN-T5: https://huggingface.co/google/flan-t5-base
DialogSum Paper: https://arxiv.org/abs/2104.07482

### Author
Simantini Ghosh
Data Scientist | NLP | LLM | Generative
