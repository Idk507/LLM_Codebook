## Finetuning Large Language Model 
Supervised Fine-Tuning
In Chapter 2 Section 2, we saw that generative language models can be fine-tuned on specific tasks like summarization and question answering. However, nowadays it is far more common to fine-tune language models on a broad range of tasks simultaneously; a method known as supervised fine-tuning (SFT). This process helps models become more versatile and capable of handling diverse use cases. Most LLMs that people interact with on platforms like ChatGPT have undergone SFT to make them more helpful and aligned with human preferences. We will separate this chapter into four sections:

1️⃣ Chat Templates
Chat templates structure interactions between users and AI models, ensuring consistent and contextually appropriate responses. They include components like system prompts and role-based messages.

2️⃣ Supervised Fine-Tuning
Supervised Fine-Tuning (SFT) is a critical process for adapting pre-trained language models to specific tasks. It involves training the model on a task-specific dataset with labeled examples. For a detailed guide on SFT, including key steps and best practices, see the supervised fine-tuning section of the TRL documentation.

3️⃣ Low Rank Adaptation (LoRA)
Low Rank Adaptation (LoRA) is a technique for fine-tuning language models by adding low-rank matrices to the model’s layers. This allows for efficient fine-tuning while preserving the model’s pre-trained knowledge. One of the key benefits of LoRA is the significant memory savings it offers, making it possible to fine-tune large models on hardware with limited resources.

4️⃣ Evaluation
Evaluation is a crucial step in the fine-tuning process. It allows us to measure the performance of the model on a task-specific dataset.

⚠️ In order to benefit from all features available with the Model Hub and 🤗 Transformers, we recommend creating an account.
References
Transformers documentation on chat templates
Script for Supervised Fine-Tuning in TRL
SFTTrainer in TRL
Direct Preference Optimization Paper
Supervised Fine-Tuning with TRL
How to fine-tune Google Gemma with ChatML and Hugging Face TRL
Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format
