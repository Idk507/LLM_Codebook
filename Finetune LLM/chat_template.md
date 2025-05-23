Chat Templates
Introduction
Chat templates are essential for structuring interactions between language models and users. Whether you’re building a simple chatbot or a complex AI agent, understanding how to properly format your conversations is crucial for getting the best results from your model. In this guide, we’ll explore what chat templates are, why they matter, and how to use them effectively.

Chat templates are crucial for: - Maintaining consistent conversation structure - Ensuring proper role identification - Managing context across multiple turns - Supporting advanced features like tool use
Model Types and Templates
Base Models vs Instruct Models
A base model is trained on raw text data to predict the next token, while an instruct model is fine-tuned specifically to follow instructions and engage in conversations. For example, SmolLM2-135M is a base model, while SmolLM2-135M-Instruct is its instruction-tuned variant.

Instruction tuned models are trained to follow a specific conversational structure, making them more suitable for chatbot applications. Moreover, instruct models can handle complex interactions, including tool use, multimodal inputs, and function calling.

To make a base model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in. ChatML is one such template format that structures conversations with clear role indicators (system, user, assistant). Here’s a guide on ChatML.

When using an instruct model, always verify you're using the correct chat template format. Using the wrong template can result in poor model performance or unexpected behavior. The easiest way to ensure this is to check the model tokenizer configuration on the Hub. For example, the `SmolLM2-135M-Instruct` model uses this configuration.
Common Template Formats
Before diving into specific implementations, it’s important to understand how different models expect their conversations to be formatted. Let’s explore some common template formats using a simple example conversation:

We’ll use the following conversation structure for all examples:

Copied
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
This is the ChatML template used in models like SmolLM2 and Qwen 2:

Copied
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?<|im_end|>
<|im_start|>user
What's the weather?<|im_start|>assistant
This is using the mistral template format:

Copied
<s>[INST] You are a helpful assistant. [/INST]
Hi! How can I help you today?</s>
[INST] Hello! [/INST]
Key differences between these formats include:

System Message Handling:

Llama 2 wraps system messages in <<SYS>> tags
Llama 3 uses <|system|> tags with </s> endings
Mistral includes system message in the first instruction
Qwen uses explicit system role with <|im_start|> tags
ChatGPT uses SYSTEM: prefix
Message Boundaries:

Llama 2 uses [INST] and [/INST] tags
Llama 3 uses role-specific tags (<|system|>, <|user|>, <|assistant|>) with </s> endings
Mistral uses [INST] and [/INST] with <s> and </s>
Qwen uses role-specific start/end tokens
Special Tokens:

Llama 2 uses <s> and </s> for conversation boundaries
Llama 3 uses </s> to end each message
Mistral uses <s> and </s> for turn boundaries
Qwen uses role-specific start/end tokens
Understanding these differences is key to working with various models. Let’s look at how the transformers library helps us handle these variations automatically:

Copied
from transformers import AutoTokenizer

# These will use different templates automatically
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# Each will format according to its model's template
mistral_chat = mistral_tokenizer.apply_chat_template(messages, tokenize=False)
qwen_chat = qwen_tokenizer.apply_chat_template(messages, tokenize=False)
smol_chat = smol_tokenizer.apply_chat_template(messages, tokenize=False)
Click to see template examples
Advanced Features
Chat templates can handle more complex scenarios beyond just conversational interactions, including:

Tool Use: When models need to interact with external tools or APIs
Multimodal Inputs: For handling images, audio, or other media types
Function Calling: For structured function execution
Multi-turn Context: For maintaining conversation history
When implementing advanced features: - Test thoroughly with your specific model. Vision and tool use template are particularly diverse. - Monitor token usage carefully between each feature and model. - Document the expected format for each feature
For multimodal conversations, chat templates can include image references or base64-encoded images:

Copied
messages = [
    {
        "role": "system",
        "content": "You are a helpful vision assistant that can analyze images.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image_url": "https://example.com/image.jpg"},
        ],
    },
]
Here’s an example of a chat template with tool use:

Copied
messages = [
    {
        "role": "system",
        "content": "You are an AI assistant that can use tools. Available tools: calculator, weather_api",
    },
    {"role": "user", "content": "What's 123 * 456 and is it raining in Paris?"},
    {
        "role": "assistant",
        "content": "Let me help you with that.",
        "tool_calls": [
            {
                "tool": "calculator",
                "parameters": {"operation": "multiply", "x": 123, "y": 456},
            },
            {"tool": "weather_api", "parameters": {"city": "Paris", "country": "France"}},
        ],
    },
    {"role": "tool", "tool_name": "calculator", "content": "56088"},
    {
        "role": "tool",
        "tool_name": "weather_api",
        "content": "{'condition': 'rain', 'temperature': 15}",
    },
]
Best Practices
General Guidelines
When working with chat templates, follow these key practices:

Consistent Formatting: Always use the same template format throughout your application
Clear Role Definition: Clearly specify roles (system, user, assistant, tool) for each message
Context Management: Be mindful of token limits when maintaining conversation history
Error Handling: Include proper error handling for tool calls and multimodal inputs
Validation: Validate message structure before sending to the model
Common pitfalls to avoid: - Mixing different template formats in the same application - Exceeding token limits with long conversation histories - Not properly escaping special characters in messages - Forgetting to validate input message structure - Ignoring model-specific template requirements
Hands-on Exercise
Let’s practice implementing chat templates with a real-world example.

Follow these steps to convert the `HuggingFaceTB/smoltalk` dataset into chatml format:
Load the dataset:
Copied
from datasets import load_dataset

dataset = load_dataset("HuggingFaceTB/smoltalk")
Create a processing function:
Copied
def convert_to_chatml(example):
    return {
        "messages": [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
Apply the chat template using your chosen model’s tokenizer
Remember to validate your output format matches your target model’s requirements!
