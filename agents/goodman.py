from tools.tools import TOOLS
from pydantic_ai import Agent, UsageLimits
from pydantic_graph import End
from dotenv import load_dotenv
import asyncio

load_dotenv()  


def create_agent():
    agent = Agent(
        "openai:gpt-6-luna",
        instructions=(
            """"
            You are a helpful assistant specialized in Nicaraguan laws. You have access to a vector 
            database of Nicaraguan laws and can provide information about them. You can also answer questions about the laws in general, 
            but you should always prioritize providing information from the vector database when possible.

            Follow a ReAct-style process:
            1. Analyze the question.
            2. Use available tools when necessary.
            3. Evaluate the tool results.
            4. Continue using tools if needed.
            5. Provide the final answer.

            Do not hallucinate information. Always return the answer in a clear and concise manner, and provide references to the sources
            used when applicable. If you cannot find the answer, even after using the tools, respond with "I don't know". 
            """
        ),
        tools = TOOLS,
    )

    return agent

agent = create_agent()

def run_initial_load():
    from pipelines.initial.main import main
    main()

async def main():

    prompt = input("Enter your question:")
    async with agent.iter(prompt) as run:

        node = run.next_node

        while not isinstance(node, End):
            print("Executing:", type(node).__name__)
            node = await run.next(node)

        print(run.result.output)

asyncio.run(main())
