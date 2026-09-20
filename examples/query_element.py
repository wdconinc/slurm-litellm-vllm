import os
import sys
from openai import OpenAI

def main():
    if len(sys.argv) < 2:
        print("Usage: python query_element.py <atomic_number>")
        sys.exit(1)
        
    atomic_number = sys.argv[1]
    
    # The OpenAI client automatically inherits OPENAI_API_BASE and OPENAI_API_KEY
    # from the environment variables sourced from endpoint.env
    client = OpenAI()
    
    # We also pull the model name from the environment, defaulting to my-local-model
    model_name = os.getenv("OPENAI_MODEL", "my-local-model")

    prompt = f"Write a brief, factual document about the chemical element with atomic number {atomic_number}. Include its name, symbol, discovery, and primary real-world uses."

    print(f"Querying model '{model_name}' for atomic number {atomic_number}...")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    
    result = response.choices[0].message.content
    
    # Save the result to a text file
    output_filename = f"element_{atomic_number}.txt"
    with open(output_filename, "w") as f:
        f.write(result)
        
    print(f"Successfully wrote {output_filename}")

if __name__ == "__main__":
    main()
