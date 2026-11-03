from pydantic import BaseModel, Field
from typing import Literal

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    char : str = Field(description="Characters of Joke")

structured_llm = llm.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a joke")

print(result)
print(result.setup)
print(result.punchline)
print(result.char)


class complaint(BaseModel):
    customer_id: int |None = Field(default= None, description="customer_id used for complaint")
    issue: str = Field(description="Issue of the customer")
    priority: Literal["low", "medium", "high", "critical"] = Field(description="Urgency of issue to be resolved, accept only low, medium, high, or critical")
    is_resolved: bool = Field(description="state of resolve answer in yes or no")

structured_llm = llm.with_structured_output(complaint)
result = structured_llm.invoke("I was charged twice for my monthly subscription. "
    "I contacted support three times, but nobody resolved it. "
    "This is urgent!")

print(result)