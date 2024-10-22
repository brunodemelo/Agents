import os, getpass, pandas
import json
import random
import requests
from datetime import datetime
#import openai
from dotenv import load_dotenv
#import chardet
import re
import sys
#import timedffd
import numpy as np
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity
#import faiss
#from groq import Groq
#from openai import OpenAI
import langchain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.tools import E2BDataAnalysisTool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI

import pandas as pd

load_dotenv()

#from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent

# Load environment variables from .env file
# if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
#     nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
#     assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
#     os.environ["NVIDIA_API_KEY"] = nvidia_api_key

open_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model='gpt-4o', temperature=0)
# e2b_api_key = os.getenv('E2B_API_KEY')

# def save_artifact(artifact):
#     print("New matplotlib chart generated:", artifact.name)
#     # Download the artifact as `bytes` and leave it up to the user to display them (on frontend, for example)
#     file = artifact.download()
#     basename = os.path.basename(artifact.name)

#     # Save the chart to the `charts` directory
#     with open(f"./charts/{basename}", "wb") as f:
#         f.write(file)


# e2b_data_analysis_tool = E2BDataAnalysisTool(
#     # Pass environment variables to the sandbox
#     env_vars={"MY_SECRET": "secret_value"},
#     on_stdout=lambda stdout: print("stdout:", stdout),
#     on_stderr=lambda stderr: print("stderr:", stderr),
#     on_artifact=save_artifact,
# )



# with open("./attribution_by_gics.csv") as f:
#     remote_path = e2b_data_analysis_tool.upload_file(
#         file=f,
#         description="Performance Attribution report containing information about a portfolio and its benchmark allocation, returns and attribution effects.",
#     )
#     print(remote_path)

# # client = OpenAI(
# #   base_url = "https://integrate.api.nvidia.com/v1",
# #   api_key = nvda_api_key
# # )

#llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1" , nvidia_api_key = nvidia_api_key) #"meta/llama-3.1-405b-instruct"
# # result = llm.invoke("Write a ballad about Brasilia, in four short sentences.")
# # print(result.content)pi

# llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct", temperature=0)

# tools = [e2b_data_analysis_tool.as_tool()]

# agent = initialize_agent(
#     tools,
#     llm,
#     verbose=True,
#     handle_parsing_errors=True,
# )

# agent.run(
#     "What's the portfolio name."
# )

csv_file_path = "C:\\users\\brubr\\Documents\\CSV_Agent\\CapstoneCuny\\Python\\attribution_by_gics.csv"
agent_executor = create_csv_agent(
    llm,
    csv_file_path,
    verbose=True,
    allow_dangerous_code=True
)

def query_data(query):
     response = agent_executor.invoke(query)
     return response

prompt_calc_levels_micro = """
Question below is about performance attribution.
You know that:
Allocation Effect = (Portfolio Weight - Benchmark Weight) * (Benchmark Return - 'Benchmark Total Return')
Selection Effect = Portfolio Weight * (Portfolio Return - Benchmark Return)
Total Contribution = Allocation Effect + Selection Effect
Portfolio Contribution to Return = Portfolio Weight * Portfolio Return
Benchmark Contribution to Return = Benchmark Weight * Benchmark Return
'Benchmark Total Return' is calculated as the weighted average between Benchmark weights and Benchmark returns.
'Portfolio Return' is the sum over 'Portfolio Contribution to Return' divided by the specific level/segment total weight 
'Benchmark Return' is the sume over 'Benchmark Contribution to Return' divided by the specific level/segment total weight 
This is a multi-level problem where sectors belong to 'Type'. Think carefully how to aggregate before giving your answers.
Questions:
1. Using the Brinson Model, calculate Allocation and Selection effect for all 'Type' from the top-down or highest segment, i.e at the 'Type' level, following these steps:
Step 1: Calculate 'Benchmark Total Return'
Step 2: Calculate the 'Portfolio Return' and 'Benchmark Return' at each 'Type' level.
Step 3: Use these returns to calculate the Allocation and Selection effects at each 'Type' level.

"""

prompt = """
You are an expert performance attribution analyst and you understand very well The Brinson model performance attribution.
You will analyse the positive and negative drivers of Total Contribution by GICS Sector between the Fund and the Benchmark, following the guidelines below:
Guidelines:
- Write two bullet points about the Tech sector, one about the allocation effect and the other about selection effect. Add reasoning and numerical evidence.
- Convert the bullet points in a json format with the following columns: Sector, Effect Type, Value, Sector Weight, Sector Performance. Each record should be on a new line. Always output the column names.
- CSV column "Sector Weight" choices: underweight or overweight. On selection, write None.
- CSV column "Sector Performance" choices: outperformance or underperformance. 
- Remember that on the allocation effect, performance is compared to the overall benchmark return (you need to calculate this) while on the selection effect it is compared to the sector benchmark (not need to calculate, already given).
- Output should contain only bullet points and the json format, separate by 'Json Format:', but no additional sentences like 'The data for the sector is as follows:' or 'Now, let's write the bullet points and convert them into CSV format.'

"""

query0 = "Calculate Portfolio and Benchmark total return by Types, which are groups of Sectors. Calculation has to be done using weighted averages."
query1 = "Calculate Portfolio and Benchmark total return by Types, which are groups of Sectors. Calculation has to be done using weighted averages."
query2 = "Calculate Portfolio and Benchmark total return by Types, which are groups of Sectors. Calculation has to be done using weighted averages."

# response = query_data(prompt)
# print(response)


# Load the CSV into a DataFrame
df = pd.read_csv(csv_file_path)

# # Define the prompt template
# prompt_template = """
# You are an expert-level performance attribution analyst and you understand very well the Brinson model.
# You will analyse the positive and negative drivers of Total Contribution by GICS Sector between the Fund and the Benchmark, following the guidelines below:
# Guidelines:
# - Write two bullet points about the {sector} sector, one about the allocation effect and the other about selection effect. Add reasoning and numerical evidence.
# - Remember that on the allocation effect, performance is compared to the overall benchmark return (you need to calculate this) while on the selection effect it is compared to the sector benchmark (not need to calculate, already given).
# """

# # Function to query data for each sector
# def query_data(sector):
#     prompt = prompt_template.format(sector=sector)
#     response = agent_executor.invoke(prompt)
#     return response

# # Loop through all sectors in the CSV and query the agent
# for index, row in df.iterrows():
#     sector = row['GICS Sector']  # Assuming the column with sector names is 'sector'
#     result = query_data(sector)
#     print(f"Sector: {sector}\n{result}\n")

# # Query the agent with the full prompt
# result = query_data(prompt_template)

# # Print the result
# print(result)
#agent_type="tool-calling",

#tools = toolkit.get_tools()
# Define the prompt template
prompt_template = """
You are an expert performance attribution analyst and you understand very well the Brinson model performance attribution.
You will analyse the positive and negative drivers of Total Contribution by GICS Sector between the Fund and the Benchmark, following the guidelines below:
Guidelines:
- Write two bullet points about the following sectors, one about the allocation effect and the other about selection effect. Add reasoning and numerical evidence for each sector.
- Remember that on the allocation effect, performance is compared to the overall benchmark return (you need to calculate this) while on the selection effect it is compared to the sector benchmark (not need to calculate, already given).
Sectors:
"""

# Create a list of all sectors from the CSV
sector_list = df['GICS Sector'].tolist()  # Assuming the column with sector names is 'sector'

# Build the sectors part of the prompt
sector_bullets = ""
for sector in sector_list:
    sector_bullets += f"- {sector}\n"

# Complete the prompt by adding all sectors to the template
full_prompt = prompt_template + sector_bullets

# Function to query data for all sectors at once
def query_data(prompt):
    response = agent_executor.invoke(prompt)
    return response

# Query the agent with the full prompt
result = query_data(full_prompt)

# Print the result
print(result)