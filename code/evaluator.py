from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Tuple
from model_loader import load_model

EVALUATION_CATEGORIES = [
    "Educational Background: Does the candidate have the appropriate educational qualifications or training for this position?",
    "Job/Organizational Fit: Has the candidate acquired similar skills or qualifications through past work experiences?",
    "Problem Solving: Did the candidate display the ability to react to a situation and devise a strategy?",
    "Verbal Communication: How were the candidate's communication skills during the interview?",
    "Candidate Interest: How much interest did the candidate show in the position and the organization?",
    "Knowledge of Organization: Did the candidate research the organization prior to the interview?",
    "Teambuilding/Interpersonal Skills: Did the candidate demonstrate, through their answers, good teambuilding/interpersonal skills?",
    "Initiative: - Did the candidate demonstrate, through their answers, a high degree of initiative?",
    "Time Management: Did the candidate demonstrate, through their answers, good time management skills?",
    "Attention to Detail: Was the candidate able to give provide examples of detail in their previous work experience?"
]

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
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def rate_response_with_llm(question: str, response: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tuple[str, str]:
    """
    Uses the LLM to rate a response on a scale of 1-5 with reasoning.
    Returns a tuple of (score, full LLM response).
    """
    messages = [{
        "role": "user",
        "content": f"""
You are a hiring manager evaluating a candidate's response to an interview question. You want to hire the best candidate for the job and are going to think critically and rate the candidate's response.

Evaluate the interview response with the following categories in mind:
{"\n".join(EVALUATION_CATEGORIES)}

You will rate the interview response on a scale from 1 to 5 based on the content of the response, using the following scale:
5 - Far Exceeds Requirements: Perfect answer. Demonstrates competency accurately, consistently, and independently. All points relevant. The answer includes only good and relavant examples.
4 - Exceeds Requirements: Demonstrates competency accurately and consistently in most situations with minimal guidance. The answer would benift from examples and many good examples provided.
3 - Meets Requirements: Demonstrates competency accurately and consistently on familiar procedures and needs supervisor guidance for new skills. The answer would benift from examples and some good examples provided.
2 - Below Requirements: Demonstrates competency inconsistently, even with repeated instruction or guidance. Although the answer would benift from examples, few good examples provided.
1 - Significant Gap: Fails to demonstrate competency regardless of guidance provided. Although the answer would benift from examples, no good examples provided.

It is extremely important that as a response you provide a clear, detailed, and conscise explanation along with the rating.
Provide your response in the following format (replace <your rating> and <your reasoning> with the values you decide):
```
Reasoning: <your reasoning>
Rating: <your rating>
```

A couple examples of responses are:
```
Reasoning: The candidate provided a clear and concise response to the question, but lacked depth in the examples they provided.
Rating: 3
```
```
Reasoning: The candidate failed to provide a clear or relevant response to the question.
Rating: 1
```
```
Reasoning: The candidate provided a detailed and thorough response to the question, with relevant examples.
Rating: 4
```

Make sure not to include the ``` character in your response as they are just used to denote the start and end of the response.

Context:
The question that was asked was: "{question}"
The interview response was: "{response}"
"""
    }]
    llm_response = generate_response(messages, tokenizer, model)
    # Extract a numerical score using determine_score below
    score = determine_score(llm_response)
    return score, llm_response

def determine_outcome_with_llm(responses: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> str:
    """
    Determines whether the candidate should advance based on the provided responses.
    """
    messages = [{
        "role": "user",
        "content": f"""
You are a hiring manager evaluating a candidate's responses to an interview. You have asked the candidate a series of questions and rated their responses for each question from 1 to 5.

Based on the following interview questions and their corresponding ratings, determine whether the candidate should:
- advance
- advance with reservations
- do not advance
Ensure that your decision is based on the candidate's overall performance across all questions. Be critical but not harsh in your evaluation and provide a clear, detailed, and concise explanation for your decision.
Provide your response in the following format:
```
Reasoning: <your reasoning>
Result: <your result>
```

A couple examples of responses are:
```
Reasoning: The candidate provided clear and concise responses to all questions, demonstrating a strong understanding of the role.
Result: advance
```
```
Reasoning: The candidate provided clear and concise responses to all questions, but lacked depth in their examples.
Result: advance with reservations
```
```
Reasoning: The candidate failed to provide a clear or relevant response to the question.
Result: do not advance
```

Make sure not to include the ``` character in your response as they are just used to denote the start and end of the response.

Context:
The questions and the scores the candidate received with their corresponding reasonings are:
{responses}
"""
    }]
    return generate_response(messages, tokenizer, model)

parse_tokenizer, parse_model, _ = load_model("TODO", use_8bit=True) # Replace "TODO" with the model ID of small to medium sized model

def determine_advance_or_not(response: str) -> str:
    """
    Determines the interview result: "Do not Advance", "Advance with reservations", or "Advance".
    """
    if "Result: " in response:
        res = response.split("Result: ")[1].split("\n")[0].strip().lower()
        if res in ["do not advance", "advance with reservations", "advance"]:
            return res
            

    messages = [{
        "role": "user",
        "content": f"""
Given the following response, determine what result of an interview the response indicates. The choices available are "do not advance", "advance with reservations", or "advance".
Return one of the three results only, with no extra explanation. If you are not able to determine the result, return "unknown".

It is very important that you only return the result (one of the previous choices or unknown) without any extra text.

Some examples of what I am looking for:
Response:
Reasoning: The candidate provided clear and concise responses to all questions, demonstrating a strong understanding of the role.
Result: advance
Final Result: advance

Response:
Reasoning: The candidate provided clear and concise responses to all questions, but lacked depth in their examples.
Result: advance with reservations
Final Result: advance with reservations

Response:
this is just some random text that doesn't contain a result
Final Result: unknown

Now, your turn.
Response:
{response}
Final Result:
"""
    }]
    return generate_response(messages, parse_tokenizer, parse_model).strip().lower()


def determine_score(response: str) -> str:
    """
    Determines the numerical score from the LLM's response.
    """
    if "Rating: " in response:
        res = response.split("Rating: ")[1].split("\n")[0].strip().lower()
        if res in ["1", "2", "3", "4", "5"]:
            return res

    messages = [{
        "role": "user",
        "content": f"""
Given the following response, determine what score/rating the response indicates on a scale from 1 to 5.
Return the number only, with no extra explanation. If you are not able to determine what score the response indicates, return -1 and nothing else.

It is very important that you only return the number without any extra text.

Some examples of what I am looking for:
Response:
Reasoning: The candidate provided a clear and concise response to the question, but lacked depth in the examples they provided.
Rating: 3
Score: 3

Response:
Reasoning: The candidate provided a detailed and thorough response to the question, with relevant examples.
Rating: 4
Score: 4

Response:
Reasoning: The candidate failed to provide a clear or relevant response to the question.
Rating: 1
Score: 1

Response:
this is just some random text that doesn't contain a rating
Score: -1

Now, your turn.
Response:
{response}
Score:
"""
    }]
    llm_score = generate_response(messages, parse_tokenizer, parse_model).strip()
    return llm_score.strip()