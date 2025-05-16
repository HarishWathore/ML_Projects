from typing import Any
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()

def main():
    print("Start...")

    ###################### Python Agent #####################################

    python_agent_instructions = """
    You are an agent designed to write and execute Python code to answer questions.
    You have access to a Python REPL, which you can use to execute Python code.
    You have the 'qrcode' package installed.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    Even if you know the answer, you must still run the code to confirm it.
    If the question cannot be answered using Python code, respond with "I don't know".
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    python_prompt = base_prompt.partial(instructions=python_agent_instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=python_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    ###################### CSV Agent #####################################

    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    ###################### Wrapper & Tool Routing #########################

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""
            Useful for generating or saving files (e.g., QR codes), executing logic, or performing tasks involving code.
            Do not include raw Python code — describe the task in natural language.
            Example: "Generate 10 QR codes pointing to www.example.com"
            """,
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""
            Useful for answering questions about the 'episode_info.csv' file.
            Provide the full natural language question.
            Example: "Which season has the most episodes?"
            """,
        ),
    ]

    grand_prompt = base_prompt.partial(instructions="""
    You are a routing agent.
    Decide which tool is best for the user query:
    - Use the Python Agent when the request involves writing or executing Python code, especially for generating QR codes, files, plots, etc.
    - Use the CSV Agent only for questions about the 'episode_info.csv' dataset.
    Always choose a tool unless the answer is completely obvious without one.
    """)

    grand_agent = create_react_agent(
        prompt=grand_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    ###################### TEST RUNS #####################################

    # Test 1: CSV Query
    print("\n---- Test 1: CSV Question ----")
    print(
        grand_agent_executor.invoke(
            {
                "input": "which season has the most episodes?",
            }
        )
    )

    # Test 2: QR Code Generation
    print("\n---- Test 2: QR Code Task ----")
    print(
        grand_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `www.linkedin.com/in/harishwathore`",
            }
        )
    )

    # Optional: Manual test of Python agent directly
    # print("\n---- Manual Test: Python Agent Directly ----")
    # print(
    #     python_agent_executor.invoke(
    #         {
    #             "input": "Generate and save in current working directory 15 qrcodes that point to `www.linkedin.com/in/harishwathore`"
    #         }
    #     )
    # )

if __name__ == "__main__":
    main()
