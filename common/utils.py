import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline as hf_pipeline
from openai import OpenAI

def load_corpus_details(corpus_path: str) -> dict:
    """
    Load corpus details into a dictionary for quick lookup.

    Args:
        corpus_path (str): Path to the corpus JSONL file.

    Returns:
        dict: A dictionary containing corpus details, keyed by NCT ID.
    """
    corpus_details = {}
    with open(corpus_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            nct_id = entry["_id"]
            metadata = entry.get("metadata", {})
            corpus_details[nct_id] = {
                "brief_title": metadata.get("brief_title", entry.get("title", "")),
                "phase": metadata.get("phase", ""),
                "drugs": metadata.get("drugs", ""),
                "drugs_list": metadata.get("drugs_list", []),
                "diseases": metadata.get("diseases", ""),
                "diseases_list": metadata.get("diseases_list", []),
                "enrollment": metadata.get("enrollment", ""),
                "inclusion_criteria": metadata.get("inclusion_criteria", ""),
                "exclusion_criteria": metadata.get("exclusion_criteria", ""),
                "brief_summary": metadata.get("brief_summary", entry.get("text", "")),
                "NCTID": nct_id
            }
    return corpus_details

def generate_keywords_gpt(client, model, messages):
    """
    Generate keywords using GPT models.

    Args:
        client (OpenAI): The OpenAI client.
        model (str): The GPT model to use.
        messages (list): The input messages for the model.

    Returns:
        str: The generated keywords as a JSON string.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip("`").strip("json")

def generate_keywords_llama(pipeline, messages):
    """
    Generate keywords using Llama models.

    Args:
        pipeline (transformers.Pipeline): The Llama model pipeline.
        messages (list): The input messages for the model.

    Returns:
        str: The generated keywords as a JSON string.
    """
    response = pipeline(messages,
                        max_new_tokens=2856,
                        do_sample=False,
                        temperature=None,
                        top_p=None)
    return response[0]['generated_text'].strip("`")

def create_llama_device_map(num_gpus, first_weight=1.0, last_weight=1.0):
    """
    Create a device map for distributing Llama model layers across multiple GPUs.

    This function calculates how to distribute the model layers across the available GPUs,
    with the option to assign different weights to the first and last GPUs.

    Args:
        num_gpus (int): The number of GPUs to use. Must be at least 2.
        first_weight (float, optional): Weight for the first GPU. Defaults to 1.0.
        last_weight (float, optional): Weight for the last GPU. Defaults to 1.0.

    Returns:
        dict: A device map specifying which GPU each model component should be assigned to.

    Raises:
        ValueError: If the number of GPUs is less than 2.

    Note:
        The function assumes a total of 80 layers in the Llama model, with special handling
        for the first and last layers, as well as additional components like embeddings and norms.
    """
    if num_gpus < 2:
        raise ValueError("Number of GPUs must be at least 2")

    # Calculate the number of layers to distribute
    distributable_layers = 77  # 79 total layers minus the first and last layers

    # Calculate weights for middle GPUs
    middle_gpus = num_gpus - 2
    middle_weight = (num_gpus - first_weight - last_weight) / middle_gpus if middle_gpus > 0 else 0

    # Calculate layers per GPU
    layers_per_gpu = [
        max(1, round(first_weight * distributable_layers / num_gpus)),  # First GPU
        *[max(1, round(middle_weight * distributable_layers / num_gpus)) for _ in range(middle_gpus)],  # Middle GPUs
        max(1, round(last_weight * distributable_layers / num_gpus))  # Last GPU
    ]

    # Adjust to ensure total is correct
    while sum(layers_per_gpu) < distributable_layers:
        layers_per_gpu[layers_per_gpu.index(min(layers_per_gpu))] += 1
    while sum(layers_per_gpu) > distributable_layers:
        layers_per_gpu[layers_per_gpu.index(max(layers_per_gpu))] -= 1

    # Create the device map
    device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
    }

    current_layer = 2
    for gpu, layer_count in enumerate(layers_per_gpu):
        for _ in range(layer_count):
            if current_layer < 78:
                device_map[f"model.layers.{current_layer}"] = gpu
                current_layer += 1

    # Assign the last layers and components to the last GPU
    last_gpu = num_gpus - 1
    device_map.update({
        "model.layers.78": last_gpu,
        "model.layers.79": last_gpu,
        "model.norm": last_gpu,
        "model.rotary_emb": last_gpu,
        "lm_head": last_gpu
    })

    return device_map


def setup_llama_pipeline(checkpoint_dir, use_quantization, num_gpus):
    """
    Set up the Llama model pipeline for text generation.

    This function configures and initializes a Llama model pipeline, optionally with
    quantization and distributed across multiple GPUs.

    Args:
        checkpoint_dir (str): The directory containing the model checkpoint.
        use_quantization (bool): Whether to use 8-bit quantization.
        num_gpus (int, optional): The number of GPUs to use for model distribution.

    Returns:
        transformers.Pipeline: The configured Llama model pipeline for text generation.

    Note:
        This function uses custom weights for the first and last GPUs in the device map
        to optimize the distribution of model components.
    """
    if use_quantization:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    #TODO: should expose these to argparse
    first_weight = .85 #0.8
    last_weight = .93# 0.6

    device_map = create_llama_device_map(int(num_gpus), first_weight, last_weight)

    llm_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    pipeline = hf_pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        device_map=device_map,
        return_full_text=False
    )
    return pipeline

def setup_model(model_name, num_gpus, checkpoint_dir=None, use_quantization=False):
    """
    Set up the model based on the model name.

    Args:
        model_name (str): The name of the model to use.
        checkpoint_dir (str, optional): The directory for Llama model checkpoints.
        use_quantization (bool): Whether to use quantization for Llama models.
        num_gpus (int, optional): The number of GPUs to use for model distribution.

    Returns:
        tuple: (model_type, model_instance)
            model_type is either 'gpt' or 'llama'
            model_instance is either an OpenAI client or a Llama pipeline
    """
    if model_name.startswith('gpt'):
        return 'gpt', OpenAI()
    else:
        return 'llama', setup_llama_pipeline(checkpoint_dir, use_quantization, num_gpus)

def generate_response(model_type, model_instance, messages, model_name=None):
    """
    Generate a response using either GPT or Llama models.

    Args:
        model_type (str): Either 'gpt' or 'llama'.
        model_instance: Either an OpenAI client or a Llama pipeline.
        messages (list): The input messages for the model.
        model_name (str, optional): The specific GPT model name (for OpenAI models).

    Returns:
        str: The generated response.
    """
    if model_type == 'gpt':
        return generate_keywords_gpt(model_instance, model_name, messages)
    else:
        return generate_keywords_llama(model_instance, messages)