from pydantic_ai import Agent, UsageLimits
from dotenv import load_dotenv

load_dotenv()  

agent = Agent(
    "openai:gpt-6-luna",
    instructions=(
        "You are a helpful Python programming tutor. "
        "Explain concepts clearly and give short examples."
    ),
)

response = agent.run("What are the best stadiums in Spain?")
print(response)