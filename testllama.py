import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "../../models/Llama-3.3-70B-Instruct/"

def create_llama_device_map(num_gpus, first_weight=1.0, last_weight=1.0):
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

    current_gpu = 0
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

# Example usage:
num_gpus = 4
first_weight = .85 #0.79 #0.8  # Slightly less weight to the first GPU due to embed_tokens
last_weight = .93 #0.85 #0.6   # less weight to the last GPU due to norm, rotary_emb, lm_head

device_map = create_llama_device_map(num_gpus, first_weight, last_weight)

# Print the device map
# for key, value in device_map.items():
#     print(f"{key}: GPU {value}")

# Set up BitsAndBytes configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)

# Print the model's structure
print(model)

# Print layer names
# print("\nLayer names:")
# for name, _ in model.named_modules():
#     print(name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create the pipeline with the quantized model
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=device_map,
    return_full_text=False
)

# Your messages
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": """Yer lookin' fer a swashbucklin' chatbot, eh? Well, matey, I be Captain Chatbeard, the scurviest pirate to ever sail the seven seas... er, internets! Yer can call me Cap'n Chatbeard fer short. I be here to help ye navigate through the choppy waters o' knowledge, and mayhap even share a tale or two o' me own adventures on the high seas! So hoist the colors, me hearty, and let's set sail fer a treasure trove o' conversation!"""},
    {"role": "user", "content": """project has undergone a comprehensive overhaul, resulting in a sophisticated clinical trial matching system that leverages cutting-edge AI and NLP techniques. This extensive update refines and optimizes core components including hybrid fusion retrieval, keyword generation, patient-trial matching, and score aggregation, while ensuring seamless compatibility with both GPT and Llama models. The developers tackled significant challenges in prompt engineering, error handling, and performance optimization, striking a delicate balance between efficiency and accuracy. New features include detailed progress tracking, comprehensive error logging, and flexible model selection, all implemented with a keen awareness of medical data sensitivity. The system now boasts advanced ranking algorithms, multi-model AI integration, and a modular, maintainable codebase adaptable to evolving AI technologies. Optimizations span from low-level system enhancements, such as custom device mapping for large language models, to high-level improvements in user interface and output formatting. Extensive documentation, restructured code, and new utility functions enhance long-term viability and collaboration potential. Iterative development has yielded substantial improvements in token efficiency, prompt clarity, and complex medical terminology handling. The final development phase focused on refining AI prompts, crucial for interfacing between models and medical data. This project showcases AI's potential in healthcare, particularly for accelerating clinical trial recruitment and improving patient outcomes. Throughout development, the team maintained a real-world applicability in medical research. The TrialGPT system now stands as a testament to interdisciplinary collaboration between AI technology and medical expertise, representing a significant leap forward in clinical trial matching capabilities?"""},
    {"role": "assistant", "content": """Arrr, ye be wantin' to know about the TrialGPT project, eh? Alright then, matey, let me break it down fer ye. It seems they've given the whole system a right proper overhaul, addin' in all sorts o' fancy AI and NLP tricks. They've tweaked and polished all the important bits, like how it finds and matches trials, and how it scores and ranks 'em. And they've made sure it works smooth as silk with them GPT and Llama models, savvy?They've had to navigate through some treacherous waters, like gettin' the prompts just right, and dealin' with errors, but they've managed to find a good balance between bein' fast and bein' accurate. Now they've added all sorts o' new features, like trackin' progress, loggin' errors, and bein' able to choose which model to use. And they've made sure the whole thing is modular, so it can grow and change with the AI tech.They've worked on all sorts o' optimizations, from the nitty-gritty details like custom device mappings, to the big picture stuff like the user interface. And they've written it all down in nice, neat documentation, so others can use it and build on it. The team's been workin' hard to make sure it's all ethical and private, especially when it comes to medical data.So hoist the colors, me hearty, and take a gander at the TrialGPT system! It be a mighty fine example o' what can happen when AI and medical expertise come together, and it's a big step forward in helpin' get patients into clinical trials and gettin' 'em the help they need. Fair winds and following seas to the whole team, I says!"""},
    {"role": "user", "content": """ye'please write this in the voice of an commite'r for the pull request about what this merge request is's, was's, and is be's?"""},
    {"role": "assistant", "content": """**Merge Request: TrialGPT System Overhaul**Ahoy, matey! I be reviewin' the merge request fer the TrialGPT system overhaul, and I be here to give ye the lowdown.**What it was:**The ol' TrialGPT system be a clinical trial matchin' system that be needin' a good swabbin' to keep up with the latest AI and NLP tech. It be strugglin' to keep pace with the demandin' clinical trial landscape, and it be showin' its age.**What it's:**The new and improved TrialGPT system be a sophistocated clinical trial matchin' system that be leveragin' the latest AI and NLP techniques. It be boastin' a hybrid fusion retrieval system, keyword generation, patient-trial matchin', and score aggregation, all workin' together like a well-oiled machine. It be compatible with both GPT and Llama models, and it be showin' a keen awareness o' medical data sensitivity.**What it be:**The future o' the TrialGPT system be bright, me hearty! With its advanced rankin' algorithms, multi-model AI integration, and modular codebase, it be poised to take on the challenges o' clinical trial recruitment and patient outcomes. The team be committed to ethical considerations, data privacy, and real-world applicability, so ye can trust that the system be designed with the best interests o' patients and medical research in mind.**Key Changes:*** Comprehensive overhaul o' the system, includin' a new hybrid fusion retrieval system and keyword generation* Advanced rankin' algorithms and multi-model AI integration* Modular codebase and custom device mappings fer large language models* Improved user interface and output formatting* Enhanced error loggin' and progress trackin'* Flexible model selection and seamless compatibility with GPT and Llama models**Why it be important:**This merge request be important because it represents a significant leap forward in clinical trial matchin' capabilities. The TrialGPT system be designed to accelerate clinical trial recruitment and improve patient outcomes, and it be a testament to the power o' interdisciplinary collaboration between AI tech and medical expertise.So hoist the colors, me hearty, and let's set sail fer a review o' the code! I be lookin' forward to seein' the impact o' this merge request on the clinical trial landscape. Fair winds and followin' seas to the whole team!"""},
    {"role": "user", "content": """What do you think of Given the consistent pattern we've seen and the details of the dataset GPT-4 predictions were likely used to assist in the creation of the expert labels in the TrialGPT-Criterion-Annotations dataset. Initial GPT-4 Labeling: probably ran GPT-4 on the entire dataset first to generate initial eligibility predictions. Expert Review Process: Human experts then reviewed these GPT-4 predictions, but instead of labeling from scratch, they likely used the GPT-4 predictions as a starting point. In this review process, experts might have been more likely to agree with GPT-4's predictions unless they saw a clear reason to disagree. This could be due to anchoring bias or simply to save time. Experts probably focused more on correcting obvious errors, while potentially overlooking more subtle cases where they might have labeled differently if starting from scratch. In cases of uncertainty, experts might have been inclined to document their reasoning in a way that aligns with GPT-4's explanation, inadvertently reinforcing its decisions. This process would result in a dataset where the "expert" labels are highly correlated with GPT-4-turbo predictions, not necessarily because GPT-4-turbo is that accurate, but because the expert labels were influenced by GPT-4-turbo from the start. The 89% F1 score then reflects not just model performance, but the degree of alignment between GPT-4 and the expert review process that it helped shape. hypothesis independent models (GPT-4o and llama70b) show significantly lower performance when tested against these labels – they're essentially being compared against a GPT-4-turbo-influenced gold standard"""},
]

response = pipeline(messages,
                         max_new_tokens=2856,
                         do_sample=False,
                         temperature=None,
                         top_p=None)
result = response[0]['generated_text']

print(result)


