import yaml
import json
import copy

import transformers
import torch
import pandas as pd

from tqdm import tqdm

def get_definition_from_word(word, cat, prompt):
    for message in prompt:
        if message["role"] == "user" and "{word}" in message["content"] and "{cat}" in message["content"]:
            message["content"] = message["content"].format(word=word, cat=cat)

    definition = get_response(prompt)
    return definition["content"]

def get_words_from_definition(definition, cat, prompt):
    for message in prompt:
        if message["role"] == "user" and "{definition}" in message["content"] and "{cat}" in message["content"]:
            message["content"] = message["content"].format(definition=definition, cat=cat)

    words = get_response(prompt)
    return words["content"]

def get_response(prompt):
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=False,
        temperature=None,
        top_k = None
    )
    return outputs[0]["generated_text"][-1]


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_name = config["model_name"]
    model_id = f"/mnt/beegfs/sinai-data/{model_name}"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Load list of words
    # with open(config["words"], "r") as f:
    #     words = f.read().splitlines()
    words = pd.read_csv(config["words"], sep="\t")

    # Load prompts
    with open(config["prompts"], "r") as f:
        prompts = json.load(f)
    
    results = pd.DataFrame(columns=["word", "pred_definition", "pred_words", "known"])

    for row in tqdm(words.itertuples(index=False)):
        print(f"WORD: {row.word}\n" + "-"*20)

        prompt = copy.deepcopy(prompts["def"])
        predicted_definition = get_definition_from_word(
            row.word, 
            row.category, 
            prompt
        )

        # Postprocess possible word leaks within the definition
        if row.word in predicted_definition:
            predicted_definition = predicted_definition.replace(row.word, "<WORD>")

        prompt = copy.deepcopy(prompts["words"])
        predicted_words = get_words_from_definition(
            predicted_definition,
            row.category,
            prompt
        )

        # Evaluate the results
        if row.word.lower() in predicted_words.lower():
            print(f"La palabra {row.word} está en la lista de palabras {predicted_words}.")
            results.loc[len(results.index)] = [row.word, predicted_definition, predicted_words, True]
        else:
            print(f"La palabra {row.word} NO está en la lista de palabras {predicted_words}.")
            results.loc[len(results.index)] = [row.word, predicted_definition, predicted_words, False]

    # Save results
    results.to_csv(f"results/{model_name}_100_palabras_DEA_it{config['iter']}.csv", sep=";", index=False)