import requests

# Together AI API configuration
TOGETHER_API_URL = "https://api.together.ai/inference"
API_KEY = "24e94eb0b1e75e2af420414a243b79dc9784ff7c70f793be99be801850a638b5"

def together_ai_inference(model_name, prompt, max_tokens=2000):  # Increased max_tokens for longer code generation
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,  # Lower temperature for more focused code generation
        "stop": ["```", "Human:", "Assistant:"]  # Stop tokens to prevent extra generation
    }
    
    response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_json = response.json()
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['text']
        else:
            raise Exception(f"Unexpected API response format: {response_json}")
    else:
        raise Exception(f"Together API request failed with status {response.status_code}: {response.text}")

def generate_rag_code(embedding_model_name="all-MiniLM-L6-v2", 
                     llm_name="meta-llama/Llama-Vision-Free", 
                     chunk_size=100):
    

    prompt =  f""""You are an expert Python developer. Create a complete implementation of a RAG (Retrieval Augmented Generation) system using the following specifications:\n\n" \
         f"Embedding Model: {embedding_model_name}\n" \
         f"Language Model: {llm_name} (accessed via Together AI API)\n" \
         f"Chunk Size: {chunk_size}\n\n" \
         "Requirements:\n" \
         "1. Include all necessary imports, including List, Dict, Any, Tuple, torch, SentenceTransformer, numpy, cosine_similarity, requests, and dataclass.\n" \
         "2. Implement text chunking with the specified chunk size, breaking down input text into manageable parts for embedding generation.\n" \
         "3. Generate embeddings using the specified embedding model.\n" \
         "4. Implement vector similarity search to find relevant chunks of text based on similarity to the query.\n" \
         "5. Initialize the LLM using the Together API with the following setup function:\n\n" \
         "```python\n" \
         "def setup_llm(api_key: str):\n" \
         "    together_llm = Together(\n" \
         "        model=\{llm_name}\",\n" \
         "        temperature=0.7,\n" \
         "        together_api_key=api_key\n" \
         "    )\n" \
         "    return together_llm\n" \
         "```\n\n" \
         "6. Ensure that the `setup_llm` function is called within the RAG pipeline.\n" \
         "7. The RAG pipeline should include a function that sends a query to the initialized Together LLM.\n" \
         "8. Include a main function demonstrating the complete RAG pipeline.\n" \
         "9. Add clear documentation and type hints throughout the code.\n" \
         "10. Implement error handling for API calls and embedding generation.\n" \
         "11. Make the code modular and reusable.\n\n" \
         "Return only the Python code without any additional explanation or markdown formatting.\n" \
         "Here's the start of the implementation:\n" \
         from typing import List, Dict, Any, Tuple\n \
         "import torch\n" \
         "from sentence_transformers import SentenceTransformer\n" \
         "import numpy as np\n" \
         "from sklearn.metrics.pairwise import cosine_similarity\n" \
         "import requests\n" \
         "from dataclasses import dataclass\n" \
         "from typing import Optional"
         "from langchain_together import Together"
f"""
    print("Generating RAG implementation code...")
    
    try:
        response = together_ai_inference(
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            prompt=prompt
        )
        return response
    except Exception as e:
        print(f"Error generating RAG code: {str(e)}")
        return None

def main():
    try:
        # Generate the RAG implementation code
        rag_code = generate_rag_code()
        
        if rag_code:
            print("\nGenerated RAG Implementation:")
            print(rag_code)
            
            # Save the generated code to a file
            with open("rag_implementation.py", "w") as f:
                f.write(rag_code)
            print("\nCode has been saved to 'rag_implementation.py'")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
