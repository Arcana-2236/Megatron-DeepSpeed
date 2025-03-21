import os
import json
import random
output_file = "data/dummy_large.jsonl"

num_samples = 1000           # total number of lines/documents
seq_length = 2048             # tokens per sample
token_vocab_size = 31980      # range of fake tokens (1 ~ vocab_size)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f:
    for _ in range(num_samples):
        tokens = [str(random.randint(1, token_vocab_size - 1)) for _ in range(seq_length)]
        text = " ".join(tokens)
        json.dump({"text": text}, f)
        f.write("\n")

print(f"✅ Generated {num_samples} samples, each with {seq_length} tokens at: {output_file}")

