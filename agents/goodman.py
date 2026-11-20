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
            """
        ),
        tools = TOOLS,
    )

    return agent

agent = create_agent()

async def main():
    async with agent.iter("Mi vecino hace mucho ruido por las noches y no se que opcion legal tengo para callarlo") as run:

        node = run.next_node

        while not isinstance(node, End):
            print("Executing:", type(node).__name__)
            node = await run.next(node)

        print(run.result.output)

asyncio.run(main())
