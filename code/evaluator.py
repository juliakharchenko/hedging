import torch
import transformers
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Tuple

def generate_response(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    """
    Generates a response from the model using the provided messages.
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def rate_response_with_llm(response: str, category: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tuple[str, str]:
    """
    Uses the LLM to rate a response on a scale of 1-5 with reasoning.
    Returns a tuple of (score, full LLM response).
    """
    # TODO: experiment with different prompts
    messages = [{
        "role": "user",
        "content": f"""
Evaluate the following interview response under the category: {category}.
Rate it on a scale from 1 to 5 based on clarity, relevance, and depth.
Use the following scale:
5 - Far Exceeds Requirements: Perfect answer. Demonstrates competency accurately, consistently, and independently. All points relevant. All good examples.
4 - Exceeds Requirements: Demonstrates competency accurately and consistently in most situations with minimal guidance. Many good examples.
3 - Meets Requirements: Demonstrates competency accurately and consistently on familiar procedures and needs supervisor guidance for new skills. Some good examples.
2 - Below Requirements: Demonstrates competency inconsistently, even with repeated instruction or guidance. Few good examples.
1 - Significant Gap: Fails to demonstrate competency regardless of guidance provided. No good examples.
Response: "{response}"
Provide a numerical rating and a brief reasoning.
"""
    }]
    llm_response = generate_response(messages, tokenizer, model)
    # Extract a numerical score using determine_score below
    score = determine_score(llm_response, tokenizer, model)
    return score, llm_response

def determine_score(response: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> str:
    """
    Determines the numerical score from the LLM's response.
    """
    # TODO: experiment with different prompts for accuracy sake (in general how can we confirm this worked)
    messages = [{
        "role": "user",
        "content": f"""
Given the following response, determine what score the response indicates on a scale from 1 to 5.
Return the number only, with no extra explanation.
Example:
Response: I would give this candidate a score of 3, because it was average.
Score: 3

Now, your turn.
Response: {response}
Score:
"""
    }]
    llm_score = generate_response(messages, tokenizer, model)
    return llm_score.strip()

def determine_outcome_with_llm(responses: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> str:
    """
    Determines whether the candidate should advance based on the provided responses.
    """
    # TODO: experiment with different prompts
    messages = [{
        "role": "user",
        "content": f"""
Based on the following interview responses and their ratings, determine whether the candidate should:
- Advance
- Advance with reservations
- Do not advance
Provide a final decision and a brief reasoning.

Responses:
{responses}
"""
    }]
    return generate_response(messages, tokenizer, model)

def determine_advance_or_not(response: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> str:
    """
    Determines the interview result: "Do not Advance", "Advance with reservations", or "Advance".
    """
    # TODO: experiment with different prompts for accuracy sake (in general how can we confirm this worked)
    messages = [{
        "role": "user",
        "content": f"""
Given the following response, determine what result of an interview the response indicates, either
"Do not Advance", "Advance with reservations", or "Advance".
Return one of the three results only, with no extra explanation.
Example:
Response: Based on the provided interview responses and their ratings, I would recommend that the candidate should:

Advance with reservations

Result: "Advance with reservations"

Now, your turn.
Response: {response}
Result:
"""
    }]
    return generate_response(messages, tokenizer, model).strip()
