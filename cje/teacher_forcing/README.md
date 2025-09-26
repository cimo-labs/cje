# CJE Teacher Forcing Module

## Overview

The teacher forcing module computes log probabilities log P(response|prompt) for importance weight calculation in CJE. It provides robust, production-ready implementations with automatic fallback mechanisms and support for various chat templates.

## When to Use

### Use **compute_teacher_forced_logprob** when:
- You need raw log P(response|prompt) for completion-style inputs
- You're working directly with the Fireworks API
- You want fine control over the computation method

### Use **compute_chat_logprob** when:
- You have chat-formatted conversations
- You need automatic template detection for Fireworks models
- You want to score assistant replies in multi-turn dialogues

### Use **Template configs** when:
- Working with specific model families (Llama, HuggingFace)
- Converting between chat and completion formats
- Ensuring correct tokenization boundaries

## File Structure

```
teacher_forcing/
├── __init__.py              # Public API exports
├── api/
│   ├── __init__.py
│   └── fireworks.py         # Fireworks API integration
├── chat.py                  # Chat conversation utilities
└── templates/               # Chat template configurations
    ├── __init__.py
    ├── base.py              # Abstract base class
    ├── fireworks.py         # Fireworks model templates
    ├── huggingface.py       # HuggingFace templates
    └── llama.py             # Llama-specific templates
```

## Core Concepts

### 1. Teacher Forcing Method
Computes log P(response|prompt) by feeding the concatenated prompt+response to the model and extracting token-level log probabilities. This avoids sampling bias from autoregressive generation.

### 2. One-Call vs Two-Call Approaches
- **One-call**: Uses byte counting to find prompt/response boundary (~89% of cases)
- **Two-call**: Fallback using difference of two API calls (100% reliability)

### 3. Chat Templates
Different models use different formatting for chat conversations. Templates handle:
- Role markers (user/assistant/system)
- Special tokens (<|begin_of_text|>, <|eot_id|>)
- Proper tokenization boundaries

## Common Interface

### Basic Teacher Forcing
```python
from cje.teacher_forcing import compute_teacher_forced_logprob

result = compute_teacher_forced_logprob(
    prompt="What is machine learning?",
    response="Machine learning is a subset of AI...",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct",
    temperature=1.0
)

if result.is_valid:
    print(f"Log probability: {result.value}")
    print(f"Method used: {result.metadata['method']}")
```

### Chat Conversations
```python
from cje.teacher_forcing import compute_chat_logprob

chat = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
]

result = compute_chat_logprob(
    chat=chat,
    model="accounts/fireworks/models/llama-v3p2-3b-instruct"
)

# Computes log P("The answer is 4." | user message + template)
```

### Custom Templates
```python
from cje.teacher_forcing import (
    HuggingFaceTemplateConfig,
    Llama3TemplateConfig,
    convert_chat_to_completions
)

# For HuggingFace models
hf_config = HuggingFaceTemplateConfig("meta-llama/Llama-3.2-3B-Instruct")

# For Llama 3 models with explicit template
llama3_config = Llama3TemplateConfig()

# Convert chat to completion format
prompt_only, prompt_plus_reply = convert_chat_to_completions(chat, hf_config)
```

## Implementation Details

### Byte Counting Algorithm
The one-call approach uses UTF-8 byte counting to find the exact boundary between prompt and response tokens:

```python
def find_boundary_by_bytes_safe(tokens, prompt, reconstructed_text):
    prompt_bytes = prompt.encode("utf-8", errors="surrogatepass")
    running = b""
    
    for idx, tok in enumerate(tokens):
        tok_bytes = tok.encode("utf-8", errors="surrogatepass")
        running += tok_bytes
        
        if len(running) == len(prompt_bytes):
            return True, idx + 1, "exact_match"
        elif len(running) > len(prompt_bytes):
            # Token spans boundary - need fallback
            return False, None, "boundary_spans_token"
```

### Two-Call Fallback
When byte counting fails (e.g., token spans boundary), the system automatically falls back to:
1. Call 1: Get log P(prompt)
2. Call 2: Get log P(prompt + response)
3. Result: log P(response|prompt) = Call 2 - Call 1

This ensures 100% reliability at the cost of an extra API call.

## Key Design Decisions

### 1. **Automatic Fallback**
Rather than failing when byte counting doesn't work, the system transparently falls back to the two-call method. This ensures reliability while optimizing for efficiency.

### 2. **Template Abstraction**
Chat templates are abstracted into configuration classes, allowing easy extension for new model families without changing core logic.

### 3. **Explicit Error Handling**
All failure modes return structured `LogProbResult` objects with clear status codes and error messages, never exceptions or magic values.

### 4. **UTF-8 Safety**
Uses `surrogatepass` error handling to deal with edge cases in tokenization, ensuring robustness with multilingual text.

### 5. **Diagnostic Metadata**
Every result includes metadata about the computation method, token counts, and failure reasons for debugging and monitoring.

## Common Issues and Solutions

### Issue: "boundary_spans_token" in metadata
**Cause**: A single token contains both prompt and response text
**Solution**: System automatically uses two-call fallback

### Issue: "echo_mismatch" error
**Cause**: API normalized whitespace or line endings differently
**Solution**: Check prompt formatting, system will use fallback

### Issue: High API latency
**Cause**: Two-call fallback doubles API requests
**Solution**: Ensure prompts don't have trailing whitespace, use shorter prompts when possible

### Issue: Template not found for model
**Cause**: Using non-Fireworks model without explicit template
**Solution**: Provide explicit `HuggingFaceTemplateConfig` or `Llama3TemplateConfig`

## Performance

### Typical Metrics
- **One-call success rate**: ~89% of requests
- **API latency**: 200-400ms (one-call), 400-800ms (two-call) 
- **Token limit**: Handles up to model's context length

### Optimization Tips
- Remove trailing whitespace from prompts
- Keep prompts under 10K characters when possible
- Reuse template configs across multiple calls
- Batch requests when computing multiple log probabilities

## Advanced Usage

### Force Two-Call Method
```python
# Skip byte counting attempt
result = compute_teacher_forced_logprob(
    prompt=prompt,
    response=response,
    model=model,
    force_two_call=True  # Always use two-call
)
```

### Custom API Configuration
```python
result = compute_teacher_forced_logprob(
    prompt=prompt,
    response=response,
    model=model,
    api_key="your-api-key",
    api_base="https://custom-endpoint.com"
)
```

### System Prompts in Chat
```python
result = compute_chat_logprob(
    chat=chat,
    model=model,
    system_prompt="You are a helpful assistant."
)
```

## Summary

The teacher forcing module provides reliable computation of log probabilities for CJE's importance weights. With automatic fallback, comprehensive template support, and production-ready error handling, it ensures accurate weight calculation across diverse models and use cases.