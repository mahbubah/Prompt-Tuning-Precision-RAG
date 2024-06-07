import os
import json
import sys
from openai import OpenAI
from math import exp
import numpy as np
#from utility.env_manager import get_env_manager
#from RAG import prompt_generator


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY") 
vectordb_keys = os.getenv("VECTORDB_MODEL") 
client = OpenAI(api_key=openai_api_key)

def get_completion(
    messages: list[dict[str, str]],
    model: str = vectordb_keys,
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    """Return the completion of the prompt.
    @parameter messages: list of dictionaries with keys 'role' and 'content'.
    @parameter model: the model to use for completion. Defaults to 'davinci'.
    @parameter max_tokens: max tokens to use for each prompt completion.
    @parameter temperature: the higher the temperature, the crazier the text
    @parameter stop: token at which text generation is stopped
    @parameter seed: random seed for text generation
    @parameter tools: list of tools to use for post-processing the output.
    @parameter logprobs: whether to return log probabilities of the output tokens or not.
    @returns completion: the completion of the prompt.
    """

    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

def file_reader(path: str, ) -> str:
    #fname = os.path.join(path)
    fname = os.path.abspath(path)
    with open(fname, 'r') as f:
        system_message = f.read()
    return system_message


'''
import os

# Specify the directory where you want to create the __init__.py file
directory_path = "/home/mahbubah/Desktop/week-7/Prompt-Tuning-Precision-RAG/RAG"

# Create the __init__.py file
init_file_path = os.path.join(directory_path, "__init__.py")
open(init_file_path, 'a').close()
'''


#env_manager = get_env_manager()
#client = OpenAI(api_key=env_manager['openai_keys']['OPENAI_API_KEY'])


def evaluate(prompt: str, user_message: str, context: str, use_test_data: bool = False) -> str:
    """Return the classification of the hallucination.
    @parameter prompt: the prompt to be completed.
    @parameter user_message: the user message to be classified.
    @parameter context: the context of the user message.
    @returns classification: the classification of the hallucination.
    """
    num_test_output = str(10)
    API_RESPONSE = get_completion(
        [
            {
                "role": "system", 
                "content": prompt.replace("{Context}", context).replace("{Question}", user_message)
            }
        ],
        model=vectordb_keys,
        logprobs=True,
        top_logprobs=1,
    )

    system_msg = str(API_RESPONSE.choices[0].message.content)

    for i, logprob in enumerate(API_RESPONSE.choices[0].logprobs.content[0].top_logprobs, start=1):
        output = f'\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob.logprob}, \naccuracy: {np.round(np.exp(logprob.logprob)*100,2)}%\n'
        print(output)
        
        if system_msg == 'true' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
            classification = 'true'
        elif system_msg == 'false' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
            classification = 'false'
        else:
            classification = 'false'

    return classification

if __name__ == "__main__":
    context_message = file_reader("../prompts/context.txt")
    prompt_message = file_reader("../prompts/prompt-generating-prompt.txt")
    context = str(context_message)
    prompt = str(prompt_message)
    
    user_message = str(input("question: "))
    
    print(evaluate(prompt, user_message, context))