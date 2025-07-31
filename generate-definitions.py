import json
import copy
import os
import yaml
import sys
import re
import torch
import stanza
import unicodedata

import pandas as pd

from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    GenerationConfig, 
    AutoTokenizer, 
    set_seed
)

def create_dataset_from_prompt(words, def_prompt):
    prompt_ds = []

    for row in tqdm(words.itertuples(index=False)):
        prompt = copy.deepcopy(def_prompt)
        
        for message in prompt:
            if message["role"] == "user" and "{word}" in message["content"] and "{cat}" in message["content"]:
                message["content"] = message["content"].replace("{word}", row.word)
                message["content"] = message["content"].replace("{cat}", row.category)
            elif message["role"] == "user" and "{word}" in message["content"]:
                message["content"] = message["content"].replace("{word}", row.word)
            else:
                pass

        prompt_ds.append(prompt)
    return prompt_ds

def generate_definitions(dataset, model, tokenizer, num_return_sequences, batch_size, stop_strings):
    # Enable mixed precision
    torch.amp.autocast("cuda", enabled=True)
    
    # Compile the model for potential speed improvements
    model = torch.compile(model)

    definitions = []

    if not stop_strings: # Quick fix for Yi problems with stop_strings parameter
        generation_config = GenerationConfig(
            max_new_tokens=128, 
            do_sample=True, 
            temperature=1.0,
            num_return_sequences=num_return_sequences,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(config["eos_token"])
            ]
        )
    else:
        generation_config = GenerationConfig(
            max_new_tokens=128, 
            do_sample=True, 
            temperature=1.0,
            num_return_sequences=num_return_sequences,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(config["eos_token"])
            ],
            stop_strings=stop_strings
        )
    # Generate definitions from words
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
        definitions += tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    
    return definitions

def process_definitions(definitions, words, num_return_sequences):
    outputs = {}
    # Postprocess possible word leaks within the definition
    for id, definition in enumerate(definitions):
        word = words.loc[id//num_return_sequences].word.lower()
        if config["categories"]:
            cat = words.loc[id//num_return_sequences].category.lower()
        definition = definition.lower()
        
        # TODO: Can we improve this word recognision?
        definition = " ".join(["<word>" if token==word else token for token in word_tokenize(definition)])
        # if word in definition:
        #     definition = definition.replace(word, "<word>")
        
        # Remove special characters from output predicted_words
        definition="".join(ch for ch in definition if ch.isalnum() or ch in [" ", "(", ")", "<", ">", "\n", ".", ","])
        definition = definition.replace("\n", " ")

        # Stop generation where special tokens appear
        special_tokens = ["<system>", "<user>", "<assistant>", "assistant"]
        for special_token in special_tokens:
            if special_token in definition:
                definition = definition.split(special_token)[0]

        outputs[id] = [word, cat, definition] if config["categories"] else [word, definition]
    return outputs

def extract_json(text):
    """ Intenta extraer JSON de un texto. """
    try:
        json_data = json.loads(text)
        if isinstance(json_data, dict) and "definición" in json_data:
            return json_data
        else:
            print(f"JSON mal formado o sin 'definición': {text}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e} - Texto: {text}")
        return None

def clean_text(text):
    """
    Remove accents and special characters from text.
    Keeps only alphanumeric characters and spaces.
    """
    # Normalize and remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Remove special characters (keep only letters, numbers and spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text

def anonymize_word_by_lemma(nlp, text, base_lemma):
    """
    Replaces all inflected forms of a word (by lemma) with the token _.

    Args:
        nlp: A Stanza pipeline.
        text (str): Input text.
        base_lemma (str): Lemma of the word to replace.

    Returns:
        str: Modified text with matching words replaced by "_".
    """
    base_lemma_norm = clean_text(base_lemma.lower())
    doc = nlp(str(text))
    output = []
    index = 0

    for sentence in doc.sentences:
        for token in sentence.tokens:
            word = token.text
            start = token.start_char
            end = token.end_char

            # Use lemma of the first word in the token (as Stanza supports multi-word tokens)
            lemma = token.words[0].lemma if token.words else ''
            lemma_norm = clean_text(lemma.lower()) if lemma else ''

            # Add untouched text between previous token and current
            output.append(text[index:start])

            # Replace word if normalized lemma matches
            if lemma_norm == base_lemma_norm:
                output.append("_")
            else:
                output.append(word)

            index = end

    output.append(text[index:])  # Add any remaining text
    return ''.join(output)

def process_definitions_v2(definitions, words, num_return_sequences, nlp):
    outputs = {}

    for id, definition in enumerate(definitions):
        word = words.loc[id // num_return_sequences].word.lower()
        print(f"Palabra: {word} - Texto original: {definition}")
        
        # Intentar extraer JSON
        definition_data = extract_json(definition)
        
        if definition_data and "definición" in definition_data:
            definition_text = definition_data["definición"]
        else:
            # Si no tiene un formato JSON correcto, intentamos limpiar la definición lo mejor posible
            definition_text = definition  # Usar el texto original si no es JSON
            definition_text = definition_text.split(":")[-1]    # Ignorar la clave del JSON si existe
            definition_text = re.sub(r'[\{\}\[\]\"]', '', definition_text).strip()

        if definition_text:
            try:
                anonymized_definition = anonymize_word_by_lemma(nlp, definition_text, word)
            except Exception as e:
                print(f"Error in lemmatization: {e}")
                anonymized_definition = re.sub(rf'\b{re.escape(word)}\b', '_', definition_text, flags=re.IGNORECASE)
        else:
            anonymized_definition = "Término no conocido."
        
        anonymized_definition = anonymized_definition.split("\n\n")[0]
        outputs[id] = [word, str(anonymized_definition)]
    
    return outputs

if __name__ == "__main__":
    set_seed(0) # for reproducibility
    
    # Read config file from arguments
    with open(sys.argv[1], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_name = config["model_name"]
    model_id = f"/mnt/beegfs/sinai-data/{model_name}"

    print(f"Generating definitions with {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    words = pd.read_csv(config["words"], sep="\t")
    words = words[:config["num_words"]]
    
    # Load prompts
    with open(config["prompts"], "r") as f:
        prompts = json.load(f)

    prompt_ds = create_dataset_from_prompt(words, prompts["def"])

    batch_size = config["batch_size_def"]
    stop_strings = config["stop_strings"]
    num_return_sequences = 5

    nlp = stanza.Pipeline(lang='es', dir='/mnt/beegfs/jcollado/stanza_resources/', processors='tokenize,mwt,pos,lemma', download_method=None)

    definitions = generate_definitions(prompt_ds, model, tokenizer, num_return_sequences, batch_size, stop_strings)
    outputs = process_definitions_v2(definitions, words, num_return_sequences, nlp=nlp)
    columns = ["word", "category", "definition"] if config["categories"] else ["word", "definition"]
    output_df = pd.DataFrame.from_dict(outputs, orient="index", columns=columns)

    # Sanitize model name
    model_name = model_name.replace("/", "_")
    
    # Save results
    outname = f"{model_name}_{config['words'].split('.')[0].split('/')[-1]}.tsv"
    outdir = f"generations/v{config['iter']}"
    if not os.path.exists(outdir):
        os.mkdir(outdir) 

    fullname = os.path.join(outdir, outname)

    output_df.to_csv(fullname, sep="\t", index=False)
