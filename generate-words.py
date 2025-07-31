import json
import copy
import os
import yaml
import sys

import pandas as pd

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    GenerationConfig, 
    AutoTokenizer, 
    set_seed
)

def create_dataset_from_prompt(definitions, words_prompt):
    prompt_ds = []

    for row in tqdm(definitions.itertuples(index=False)):
        prompt = copy.deepcopy(words_prompt)
        
        for message in prompt:
            if message["role"] == "user" and "{definition}" in message["content"] and "{cat}" in message["content"]:
                message["content"] = message["content"].replace("{definition}", row.definition)
                message["content"] = message["content"].replace("{cat}", row.category)
            elif message["role"] == "user" and "{definition}" in message["content"]:
                message["content"] = message["content"].replace("{definition}", row.definition)
            else:
                pass

        prompt_ds.append(prompt)
    return prompt_ds

def generate_words(dataset, model, tokenizer, batch_size, stop_strings):
    words = []

    if stop_strings:
        generation_config = GenerationConfig(
            max_new_tokens=128, 
            do_sample=False,
            stop_strings=stop_strings
        )
    else:
        generation_config = GenerationConfig(
            max_new_tokens=128, 
            do_sample=False
        )  

    # Generate words from definitions
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]

        model_inputs = tokenizer.apply_chat_template(
            batch, 
            add_generation_prompt=True, 
            padding=True, 
            truncation=False,
            return_tensors="pt"
        ).to("cuda")

        input_length = model_inputs.shape[1] # Get input_length to remove the whole input from the decoding output
        outputs = model.generate(model_inputs, generation_config=generation_config, tokenizer=tokenizer)
        words += tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    
    return words

def extract_json(text):
    """ Intenta extraer JSON de un texto. """
    try:
        json_data = json.loads(text)
        if isinstance(json_data, dict) and "palabras" in json_data:
            return json_data
        else:
            print(f"JSON mal formado o sin 'palabras': {text}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e} - Texto: {text}")
        return None
    
def process_words(words, definitions):
    outputs = {}
    # Postprocess possible word leaks within the definition
    for id, predicted_words in enumerate(words):
        word = definitions.loc[id].word.lower()

        if config["categories"]:
            cat = definitions.loc[id].category.lower()
        definition = definitions.loc[id].definition.lower()
        predicted_words = predicted_words.lower()

        print(f"Palabra: {word} - Texto original: {predicted_words}")
        
        # Intentar extraer JSON
        predicted_words_data = extract_json(predicted_words)

        if predicted_words_data and "palabras" in predicted_words_data:
            predicted_words_text = predicted_words_data["palabras"]
            predicted_words_text = " ".join(map(str, predicted_words_text))
        else:
            # Si no tiene un formato JSON correcto, intentamos limpiar la definición lo mejor posible
            predicted_words_text = predicted_words  # Usar el texto original si no es JSON
            predicted_words_text = predicted_words_text.split(":")[-1]    # Ignorar la clave del JSON si existe
            predicted_words_text = predicted_words_text.replace('"', '') # Eliminar posibles comillas dobles sobrantes
            predicted_words_text = predicted_words_text.replace("'", '') # Eliminar posibles comillas simples sobrantes
            predicted_words_text = predicted_words_text.replace('{', '') # Eliminar posibles llaves sobrantes
            predicted_words_text = predicted_words_text.replace('}', '') # Eliminar posibles llaves sobrantes
            predicted_words_text = predicted_words_text.replace('[', '') # Eliminar posibles corchetes sobrantes
            predicted_words_text = predicted_words_text.replace(']', '') # Eliminar posibles corchetes sobrantes
            predicted_words_text = predicted_words_text.strip()   # Eliminar posibles espacios sobrantes
        
        if not predicted_words_text:
            predicted_words_text = "Definición no conocida."

        print(f"Texto final: {predicted_words_text}")
        
        outputs[id] = [word, cat, definition, predicted_words_text] if config["categories"] else [word, definition, predicted_words_text]
    return outputs
    
if __name__ == "__main__":
    set_seed(0)

    with open(sys.argv[1], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_name = config["model_name"]
    model_id = f"/mnt/beegfs/sinai-data/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    # Sanitize model name
    model_name = model_name.replace("/", "_")
    definitions_path = f"generations/v{config['iter']}/{model_name}_{config['words'].split('.')[0].split('/')[-1]}.tsv"
    definitions = pd.read_csv(definitions_path, sep="\t", keep_default_na=False, na_values=[])

    print(f"Generating list of words with {model_name}")

    # Load prompts
    with open(config["prompts"], "r") as f:
        prompts = json.load(f)

    prompt_ds = create_dataset_from_prompt(definitions, prompts["words"])

    batch_size = config["batch_size_words"]
    stop_strings = config["stop_strings"]

    words = generate_words(prompt_ds, model, tokenizer, batch_size, stop_strings)
    outputs = process_words(words, definitions)
    columns = ["word", "category", "definition", "predicted_words"] if config["categories"] else ["word", "definition", "predicted_words"]
    output_df = pd.DataFrame.from_dict(outputs, orient="index", columns=columns)
    
    # Save results
    outname = f"{model_name}_{config['words'].split('.')[0].split('/')[-1]}.tsv"
    outdir = f"generations/v{config['iter']}"
    if not os.path.exists(outdir):
        os.mkdir(outdir) 

    fullname = os.path.join(outdir, outname)

    output_df.to_csv(fullname, sep="\t", index=False)
