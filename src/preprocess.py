import json
import pandas as pd
import os

input_path = os.path.join("data", "problems_data.jsonl")
output_path = os.path.join("data", "problems.csv")

rows = []

print("📂 Opening JSONL file:", input_path)

with open(input_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue

        item = json.loads(line)

        title = item.get("title", "")
        description = item.get("description", "")
        input_desc = item.get("input_description", "")
        output_desc = item.get("output_description", "")
        problem_class = item.get("problem_class", "")
        problem_score = item.get("problem_score", "")

        text = f"{title} {description} {input_desc} {output_desc}"

        rows.append({
            "text": text,
            "title": title,
            "description": description,
            "input_description": input_desc,
            "output_description": output_desc,
            "problem_class": problem_class,
            "problem_score": problem_score
        })

        if idx % 500 == 0 and idx > 0:
            print(f"Processed {idx} lines")

print("✅ Total rows extracted:", len(rows))

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False, encoding="utf-8")

print("🎉 problems.csv successfully written!")
print(df.head(2))
