import yaml
import os
import sys

import pandas as pd
from nltk.tokenize import word_tokenize

with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_name = config["model_name"]

# Sanitize model name
model_name = model_name.replace("/", "_")

print(f"Evaluating list of words with {model_name}")

definitions_path = f"generations/v{config['iter']}/{model_name}_{config['words'].split('.')[0].split('/')[-1]}.tsv"
df = pd.read_csv(definitions_path, sep="\t")

df = df.fillna("") # Make sure all cells have valid values

# Function to check if the word exists in the predicted_words
df['known'] = df.apply(lambda row: row['word'] in word_tokenize(row['predicted_words']), axis=1)

# Group every 5 rows and check if any 'known' is True in the group
group_known = df.groupby(df.index // 5)['known'].transform('any')

# Select one word from each group (since each group contains the same word)
unique_words = df.groupby(df.index // 5)['word'].first()

# Create the new DataFrame
result_df = pd.DataFrame({
    'word': unique_words,
    'group_known': group_known[::5].values  # Take one value for each group
})

# Save results
outname = f"{model_name}_{config['words'].split('.')[0].split('/')[-1]}.tsv"
outdir = f"results/v{config['iter']}"
if not os.path.exists(outdir):
    os.mkdir(outdir) 

fullname = os.path.join(outdir, outname)

result_df.to_csv(fullname, sep="\t", index=False)