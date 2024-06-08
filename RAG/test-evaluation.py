import requests
import os
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter  
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    #context_utilization,
)

from dotenv import load_dotenv,find_dotenv


# Data loader
def chunk_loader(file_path= '../prompts/context.txt'):
    loader = TextLoader(file_path)
    documents = loader.load()

    # Chunk the data
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_retriever(chunks):
  # Load OpenAI API key from .env file
  load_dotenv(find_dotenv())

  # Setup vector database
  client = weaviate.Client(
    embedded_options = EmbeddedOptions()
  )

  # Populate vector database
  vectorstore = Weaviate.from_documents(
      client = client,    
      documents = chunks,
      embedding = OpenAIEmbeddings(),
      by_text = False
  )

  # Define vectorstore as retriever to enable semantic search
  retriever = vectorstore.as_retriever()
  return retriever


def file_reader(path: str, ) -> str:
    fname = os.path.join(path)
    with open(fname, 'r') as f:
        system_message = f.read()
    return system_message

import json

def file_reader_json(path: str) -> list:
    fname = os.path.join(path)
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def test_prompts():
    prompts = file_reader("../prompts/automatically-generated-prompts.txt")
    chunks =  chunk_loader()
    retriever = create_retriever(chunks)

    # Define LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

    final_prompts = []

    for prompt in prompts:
       final_prompts.append(ChatPromptTemplate.from_template(prompt))

    # prompt = ChatPromptTemplate.from_template(template)

    for prompt in final_prompts:
        # Setup RAG pipeline
        rag_chain = (
            {"context": retriever,  "question": RunnablePassthrough()} 
            | prompt 
            | llm
            | StrOutputParser() 
        )

        test_cases = file_reader_json("../prompts/automatically-generated-test-prompts.txt")

        questions = []
        ground_truths = []
        for test_case in test_cases:
            questions.append(test_case["user"])
            ground_truths.append(test_case["assistant"])

        answers = []
        contexts = []

        # Inference
        for query in questions:
            answers.append(rag_chain.invoke(query))
            contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

        # To dict
        '''data = {
            "question": questions, # list 
            "answer": answers, # list
            "contexts": contexts, # list list
            "ground_truths": ground_truths # list Lists
        }'''


# Combine questions, answers, contexts, and ground_truths into a list of dictionaries
    data = []
    for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths):
        data.append({
        "question": question,
        "answer": answer,
        "contexts": context,
        "ground_truth": ground_truth  # Add ground_truth column
    })

        # Convert dict to dataset
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset = dataset, 
            metrics=[
                #context_utilization,
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )

        return result
    
if __name__ == "__main__":
    test_prompts()