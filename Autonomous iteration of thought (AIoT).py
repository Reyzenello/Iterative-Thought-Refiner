import requests
import json

# Define the LLMA (LLM Agent)
def LLMA(query, prompt, knowledge_base):
    """
    Calls the Ollama API to process prompts and generate responses.
    """
    # Combine query and prompt
    full_prompt = f"{prompt}\n\nQuestion: {query}"

    # Call the Ollama API
    response = ollama_query("llama3.1", full_prompt)
    return response

# Define the IDA (Inner Dialogue Agent)
def IDA(query, last_response):
    """
    Generates a new prompt based on the query and the last response.
    """
    # Generate a new prompt to refine the previous answer
    new_prompt = f"Please refine your previous answer: '{last_response}' considering any missing details for the question: '{query}'."
    return new_prompt

# Define the stopping criterion
def stopping_criterion(response, config=None):
    """
    Determines whether to stop the iteration.
    """
    # In AIoT, the LLMA decides when to stop by including a specific token or phrase.
    return "Final Answer" in response

# Autonomous Iteration of Thought (AIoT)
def AIoT_algorithm(query, max_iterations, knowledge_base):
    """
    Implements the AIoT algorithm where the LLMA decides when to stop iterating.
    """
    # Initial prompt
    prompt = "Provide a detailed answer to the following question."
    response = LLMA(query, prompt, knowledge_base)
    iteration_stop = stopping_criterion(response)
    iteration = 1

    # Iterative loop
    while not iteration_stop and iteration <= max_iterations:
        # IDA generates a new prompt
        prompt = IDA(query, response)
        # LLMA generates a new response
        response = LLMA(query, prompt, knowledge_base)
        # Check stopping criterion
        iteration_stop = stopping_criterion(response)
        iteration += 1

    return response

# Guided Iteration of Thought (GIoT)
def GIoT_algorithm(query, iterations, knowledge_base):
    """
    Implements the GIoT algorithm where the number of iterations is fixed.
    """
    # Initial prompt
    prompt = "Provide a detailed answer to the following question."
    response = LLMA(query, prompt, knowledge_base)

    for iteration in range(1, iterations):
        # IDA generates a new prompt
        prompt = IDA(query, response)
        # LLMA generates a new response
        response = LLMA(query, prompt, knowledge_base)

    # Final iteration with explicit final instructions
    prompt = IDA(query, response) + "\n\nPlease provide your final answer."
    response = LLMA(query, prompt, knowledge_base)
    return response

# Ollama Query Function
def ollama_query(model, prompt):
    """
    Calls the Ollama API with the given model and prompt.
    """
    url = "http://127.0.0.1:11434/api/generate"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt
    }

    # Set stream=True to handle streaming responses
    response = requests.post(url, json=payload, headers=headers, stream=True)
    result = ""
    # Iterate over the streaming response line by line
    for line in response.iter_lines():
        if line:
            line_content = line.decode('utf-8')
            try:
                # Parse each line as a JSON object
                json_data = json.loads(line_content)
                if 'response' in json_data:
                    # Accumulate the 'response' field
                    result += json_data['response']
            except json.JSONDecodeError:
                print(f"Failed to decode JSON: {line_content}")
                pass
    # Return the accumulated result
    return result if result else "Error generating response."

# Example usage
if __name__ == "__main__":
    query = "How many r are present in the word Raspberry?"
    knowledge_base = {"basic_info": "general knowledge"}
    max_iterations = 5  # For AIoT
    fixed_iterations = 3  # For GIoT

    # Using AIoT
    print("=== Autonomous Iteration of Thought (AIoT) ===")
    final_response_aiot = AIoT_algorithm(query, max_iterations, knowledge_base)
    print(f"Final response (AIoT): {final_response_aiot}\n")

    # Using GIoT
    print("=== Guided Iteration of Thought (GIoT) ===")
    final_response_giot = GIoT_algorithm(query, fixed_iterations, knowledge_base)
    print(f"Final response (GIoT): {final_response_giot}")
