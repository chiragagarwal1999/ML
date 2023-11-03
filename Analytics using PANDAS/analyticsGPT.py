#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

# # GPU Config
# # Added a paramater for GPU layer numbers
# n_gpu_layers = os.environ.get('N_GPU_LAYERS')
#
# # Added custom directory path for CUDA dynamic library
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/extras/CUPTI/lib64")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include")
# os.add_dll_directory("C:/tools/cuda/bin")

from constants import CHROMA_SETTINGS

def main():
    args = parse_arguments()
   
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Prepare the LLM
    #if model_type == "LlamaCpp":
    #        llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False, n_gpu_layers=n_gpu_layers)
    #elif model_type == "GPT4All":
    #        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    #else:
            # raise exception if model_type is not supported
    #        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    df = pd.read_csv('C:/project/privateGPT/privateGPT/source_documents/employees.csv') 

    # Initializing the agent 
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), 
              df, verbose=True, handle_parsing_errors="Check your output and make sure it conforms!") 

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        agent.run(query)
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
