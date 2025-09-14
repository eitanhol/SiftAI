import json
import pandas as pd

with open("data_result.json", "r") as f:
    raw = json.load(f)

# Access only the "data" section
data = raw["data"]
print(data)
# Extract column names
columns = [
    col[key] 
    for col in data["columns"] 
    for key in col 
    if "name" in key.lower()
]


# Create DataFrame from rows
df = pd.DataFrame(data["rows"], columns=columns)
df.to_csv("data1.csv", index=False)

html_table = df.to_html(classes="table table-striped", index=False)

print(df.head())   # show first 5 rows