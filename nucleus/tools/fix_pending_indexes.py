import os

STUBS = [
    "docs/deployment/index.md",
    "docs/mcp/index.md",
    "docs/membrane/index.md",
    "docs/reference/index.md"
]

def generate_index(path):
    dirname = os.path.dirname(path)
    title = os.path.basename(dirname).title().replace("-", " ")
    
    entries = []
    
    # List directories
    for item in sorted(os.listdir(dirname)):
        full_path = os.path.join(dirname, item)
        if os.path.isdir(full_path):
            if item in ["assets", "images", "includes"]: continue
            # Check if dir has index
            if os.path.exists(os.path.join(full_path, "index.md")):
                entries.append(f"- [**{item.title()}**]({item}/index.md)")
            else:
                entries.append(f"- [**{item.title()}**]({item}/)")
        elif item.endswith(".md") and item != "index.md":
            name = item.replace(".md", "").replace("-", " ").title()
            entries.append(f"- [{name}]({item})")

    content = f"""# {title}

## Overview

Index of {title} resources.

## Contents

{chr(10).join(entries)}
"""
    return content

for stub in STUBS:
    if os.path.exists(stub):
        print(f"Fixing {stub}...")
        content = generate_index(stub)
        with open(stub, "w") as f:
            f.write(content)

print("Done.")
