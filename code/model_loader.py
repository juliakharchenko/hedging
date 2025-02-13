import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.pipelines import Pipeline

def get_device_config() -> dict[str, object]:
    """
    Determines the available device and returns a dictionary with the appropriate
    'device_map' and 'torch_dtype' settings.
    """
    if torch.cuda.is_available():
        return {"device_map": "cuda", "torch_dtype": torch.bfloat16}
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # mps does not currently support bfloat16 well, so use float16 instead
        return {"device_map": "mps", "torch_dtype": torch.float16}
    else:
        return {"device_map": "cpu", "torch_dtype": torch.float32}

def load_model(model_id: str, use_8bit: bool = True) -> tuple[AutoTokenizer, AutoModelForCausalLM, Pipeline]:
    """
    Loads the tokenizer, model, and pipeline for a given model_id.
    If use_8bit is True, uses quantization with BitsAndBytesConfig.
    The device and data type are determined dynamically.
    """
    device_config = get_device_config()
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if bnb_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            **device_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **device_config)
    
    # Create a text-generation pipeline (if needed elsewhere)
    gen_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **device_config,
    )
    
    return tokenizer, model, gen_pipeline
