from mistralai import Mistral
import os, re
import json
import random
import requests
from datetime import datetime
#import openai
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv('MISTRAL_API_KEY')


client = Mistral(api_key=api_key)



planning_agent_id =  "ag:a29b7af4:20241021:planning-agent:f6b0060e"
summarization_agent_id = "ag:a29b7af4:20241021:summarization-agent:526aed65"
python_agent_id = "ag:a29b7af4:20241021:python-agent:0ca4d9e1"
performance_agent_id = "ag:a29b7af4:20241021:performance-attribution-planning-agent:f66418dc"
reporting_agent_id = "ag:a29b7af4:20241021:reporting-agent:151691e2"

def run_analysis_planning_agent(query):
    """
    Sends a user query to a Python agent and returns the response.

    Args:
        query (str): The user query to be sent to the Python agent.

    Returns:
        str: The response content from the Python agent.
    """
    print("### Run Planning agent")
    print(f"User query: {query}")
    try:
        response = client.agents.complete(
            agent_id= performance_agent_id,
            messages = [
                {
                    "role": "user",
                    "content":  query
                },
            ]
        )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(f"Request failed: {e}. Please check your request.")
        return None
    

    

# query = """
# Load this data: https://raw.githubusercontent.com/fivethirtyeight/data/master/bad-drivers/bad-drivers.csv

# The dataset consists of 51 datapoints and has eight columns:
# - State
# - Number of drivers involved in fatal collisions per billion miles
# - Percentage Of Drivers Involved In Fatal Collisions Who Were Speeding
# - Percentage Of Drivers Involved In Fatal Collisions Who Were Alcohol-Impaired
# - Percentage Of Drivers Involved In Fatal Collisions Who Were Not Distracted
# - Percentage Of Drivers Involved In Fatal Collisions Who Had Not Been Involved In Any Previous Accidents
# - Car Insurance Premiums 
# """

query = """
Load this data: https://raw.githubusercontent.com/brunodemelo/Agents/refs/heads/main/attribution_by_gics.csv

The dataset consists of the following columns, which are typically found in a performance attribution report: 
Type,GICS Sector,Portfolio Weight,Benchmark Weight,Portfolio Return,Benchmark Return,Variation in Weight,Variation in Return,Allocation Effect,Selection Effect,Total Contribution

"""

planning_result = run_analysis_planning_agent(query)

print(planning_result)

class PythonAgentWorkflow:
    def __init__(self):
        pass

    def extract_pattern(self, text, pattern):
        """
        Extracts a pattern from the given text.

        Args:
            text (str): The text to search within.
            pattern (str): The regex pattern to search for.

        Returns:
            str: The extracted pattern or None if not found.
        """
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_step_i(self, planning_result, i, n_step):
        """
        Extracts the content of a specific step from the planning result.

        Args:
            planning_result (str): The planning result text.
            i (int): The step number to extract.
            n_step (int): The total number of steps.

        Returns:
            str: The extracted step content or None if not found.
        """
        if i < n_step:
            pattern = rf'## Step {i}:(.*?)## Step {i+1}'
        elif i == n_step:
            pattern = rf'## Step {i}:(.*)'
        else:
            print(f"Invalid step number {i}. It should be between 1 and {n_step}.")
            return None

        step_i = self.extract_pattern(planning_result, pattern)
        if not step_i:
            print(f"Failed to extract Step {i} content.")
            return None

        return step_i

    def extract_code(self, python_agent_result):
          """
          Extracts Python function and test case from the response content.

          Args:
              result (str): The response content from the Python agent.

          Returns:
              tuple: A tuple containing the extracted Python function and a retry flag.
          """
          retry = False
          print("### Extracting Python code")
          python_code = self.extract_pattern(python_agent_result, r'```python(.*?)```')
          if not python_code:
              retry = True
              print("Python function failed to generate or wrong output format. Setting retry to True.")

          return python_code, retry

    def run_python_agent(self, query):
        """
        Sends a user query to a Python agent and returns the response.

        Args:
            query (str): The user query to be sent to the Python agent.

        Returns:
            str: The response content from the Python agent.
        """
        print("### Run Python agent")
        print(f"User query: {query}")
        try:
            response = client.agents.complete(
                agent_id= python_agent_id,
                messages = [
                    {
                        "role": "user",
                        "content":  query
                    },
                ]
            )
            result = response.choices[0].message.content
            return result

        except Exception as e:
            print(f"Request failed: {e}. Please check your request.")
            return None

    def check_code(self, python_function, state):
        """
        Executes the Python function and checks for any errors.

        Args:
            python_function (str): The Python function to be executed.

        Returns:
            bool: A flag indicating whether the code execution needs to be retried.

        Warning:
            This code is designed to run code that’s been generated by a model, which may not be entirely reliable.
            It's strongly recommended to run this in a sandbox environment.
        """
        retry = False
        try:
            print(f"### Python function to run: {python_function}")
            exec(python_function, state)
            print("Code executed successfully.")
        except Exception:
            print(f"Code failed.")
            retry = True
            print("Setting retry to True")
        return retry

    def process_step(self, planning_result, i, n_step, max_retries, state):
        """
        Processes a single step, including retries.

        Args:
            planning_result (str): The planning result text.
            i (int): The step number to process.
            n_step (int): The total number of steps.
            max_retries (int): The maximum number of retries.

        Returns:
            str: The extracted step content or None if not found.
        """

        retry = True
        j = 0
        while j < max_retries and retry:
            print(f"TRY # {j}")
            j += 1
            step_i = self.extract_step_i(planning_result, i, n_step)
            if step_i:
                print(step_i)
                python_agent_result = self.run_python_agent(step_i)
                python_code, retry = self.extract_code(python_agent_result)
                print(python_code)
                retry = self.check_code(python_code, state)
        return None

    def workflow(self, planning_result):
        """
        Executes the workflow for processing planning results.

        Args:
            planning_result (str): The planning result text.
        """
        state = {}
        print("### ENTER WORKFLOW")
        n_step = int(self.extract_pattern(planning_result, '## Total number of steps:\s*(\d+)'))
        for i in range(1, n_step + 1):
            print(f"STEP # {i}")
            self.process_step(planning_result, i, n_step, max_retries=2, state=state)


        print("### Exit WORKFLOW")
        return None

import sys
import io

# See the output of print statements in the console while also capturing it in a variable,
class Tee(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_stdout = sys.stdout

    def write(self, data):
        self.original_stdout.write(data)
        super().write(data)

    def flush(self):
        self.original_stdout.flush()
        super().flush()

# Create an instance of the Tee class
tee_stream = Tee()

# Redirect stdout to the Tee instance
sys.stdout = tee_stream


Python_agent = PythonAgentWorkflow()
Python_agent.workflow(planning_result)

# Restore the original stdout
sys.stdout = tee_stream.original_stdout

# Get the captured output
captured_output = tee_stream.getvalue()

response = client.agents.complete(
    agent_id= reporting_agent_id,
    messages = [
        {
            "role": "user",
            "content":  query + captured_output
        },
    ]
)
result = response.choices[0].message.content
print(result)