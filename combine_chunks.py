import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--glob_template", type=str, required=True)
args = parser.parse_args()

root = args.root
glob_template = args.glob_template
chunks = list( glob.glob(f'{root}/{glob_template}'))
print(f"Found {len(chunks)} chunks")

for chunk in chunks:
    print(chunk)

# Combine all chunks into a single dataframe
combined_df = []

for chunk in chunks:
    df = pd.read_csv(chunk)
    combined_df.append(df)

# Save the combined dataframe
combined_df = pd.concat(combined_df)
combined_df.to_csv( root / (glob_template[:-4] + "_combined.csv"), index=False)