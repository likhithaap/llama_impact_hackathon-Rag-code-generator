import requests

# Together AI API configuration
TOGETHER_API_URL = "https://api.together.ai/inference"
API_KEY = "fbfb3084148fa76a6dcfb97852ef52321779645e9ad32206aaf4e638e3ca406b"

def together_ai_inference(model_name, prompt, max_tokens=2000):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    
    response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        response_json = response.json()
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['text']
    return None

def generate_rag_code(embedding_model_name="all-MiniLM-L6-v2", 
                     llm_name="meta-llama/Llama-Vision-Free", 
                     chunk_size=100):
    
    prompt = f"""
    You are an expert Python developer. Create a complete implementation of a RAG (Retrieval Augmented Generation) system using the following specifications:

    Embedding Model: {embedding_model_name}
    Language Model: {llm_name} (accessed via Together AI API)
    Chunk Size: {chunk_size}

    Requirements:
    1. Include necessary imports (List, Dict, Any, Tuple, torch, SentenceTransformer, numpy, cosine_similarity, requests, dataclass, PyPDF2).
    2. Implement a PDF document processor that:
       - Extracts text from PDF files
       - Handles multiple PDF documents
       - Processes PDF metadata
       - Handles PDF reading errors gracefully
    3. Implement text chunking with the specified chunk size.
    4. Generate embeddings using the specified embedding model.
    5. Implement vector similarity search for retrieving relevant text chunks.
    6. Initialize the LLM via Together API with the following setup function:

    ```python
    def setup_llm(api_key: str):
        together_llm = Together(
            model="{llm_name}",
            temperature=0.7,
            together_api_key=api_key
        )
        return together_llm
    ```

    7. Ensure `setup_llm` is called within the RAG pipeline.
    8. The RAG pipeline should include:
       - PDF document loading and processing
       - Text extraction and chunking
       - Embedding generation
       - Query processing
       - Response generation
    9. Include a main function demonstrating the complete RAG pipeline with PDF document processing.
    10. Add clear documentation and type hints throughout the code.
    11. Implement error handling for PDF processing errors, API calls, and embedding generation errors.
    12. Make the code modular and reusable.
    13. Include a PDFProcessor class with methods for loading PDFs, extracting text, and processing metadata.
    
    Return only the Python code without additional explanation or markdown formatting.
    """
    
    print("Generating RAG implementation code with PDF support...")
    
    try:
        response = together_ai_inference(
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            prompt=prompt
        )
        return response
    except Exception as e:
        print(f"Error generating RAG code: {str(e)}")
        return None

def evaluate_code(code: str) -> dict:
    """
    Use LLM to evaluate the code for potential issues and best practices.
    Returns a dictionary containing evaluation results and suggested improvements.
    """
    evaluation_prompt = f"""
    As an expert Python developer, evaluate the following code for potential issues, 
    focusing on:
    1. Syntax errors
    2. Logic errors
    3. Best practices violations
    4. Missing error handling
    5. Import issues
    6. Type hint completeness
    7. Documentation quality
    8. Code structure and modularity
    9. Memory efficiency
    10. API usage correctness

    Provide your evaluation in the following JSON format:
    {{
        "has_errors": boolean,
        "critical_issues": [list of critical issues found],
        "warnings": [list of non-critical issues],
        "suggestions": [list of improvement suggestions],
        "corrected_code": "full corrected code if there are critical issues"
    }}

    Code to evaluate:

    {code}
    """
    
    response = together_ai_inference(
        model_name="codellama/CodeLlama-34b-Python-hf",
        prompt=evaluation_prompt
    )
    
    try:
        # Note: In practice, you'd want to properly parse the JSON response
        # Here we're assuming the LLM returns properly formatted JSON
        return eval(response)
    except Exception as e:
        return {
            "has_errors": True,
            "critical_issues": [f"Failed to parse evaluation response: {str(e)}"],
            "warnings": [],
            "suggestions": [],
            "corrected_code": None
        }

def generate_and_evaluate_rag_code():
    # Generate the initial code
    rag_code = generate_rag_code()
    with open("rag_implementation_with_pdf.py", "w") as f:
        f.write(rag_code)
    print("Corrected code has been saved to rag_implementation_with_pdf.py")
    
    if rag_code:
        print("\nEvaluating generated code...")
        evaluation_results = evaluate_code(rag_code)
        
        if evaluation_results["has_errors"]:
            print("\nCritical issues found:")
            for issue in evaluation_results["critical_issues"]:
                print(f"- {issue}")
            
            if evaluation_results["corrected_code"]:
                print("\nSuggested corrections have been generated.")
                corrected_code = evaluation_results["corrected_code"]
                
                # Save the corrected code
                with open("rag_implementation_with_pdf.py", "w") as f:
                    f.write(corrected_code)
                print("Corrected code has been saved to rag_implementation_with_pdf.py")
                
                # Re-evaluate the corrected code
                print("\nRe-evaluating corrected code...")
                final_evaluation = evaluate_code(corrected_code)
                if not final_evaluation["has_errors"]:
                    print("The corrected code passes evaluation.")
                else:
                    print("The corrected code still has some issues that may need manual review.")
            else:
                print("No automatic corrections could be generated. Manual review required.")
        else:
            print("No critical issues found in the generated code.")
        
        if evaluation_results["warnings"]:
            print("\nWarnings:")
            for warning in evaluation_results["warnings"]:
                print(f"- {warning}")
        
        if evaluation_results["suggestions"]:
            print("\nSuggestions for improvement:")
            for suggestion in evaluation_results["suggestions"]:
                print(f"- {suggestion}")
    else:
        print("Failed to generate initial RAG code.")
    

if __name__ == "__main__":
    generate_and_evaluate_rag_code()