from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0)

"""messages = [
    SystemMessage(
        content="You are a professional Data Quality Analyst."
    ),
    HumanMessage(
        content="Identify issues in customer_id=101, email=NULL."
    )
]



messages = [
    SystemMessage(content="You are a professional Data Engineer."),
    HumanMessage(content="explain Duplicate Records.")
            ]

response = llm.invoke(messages)

print(response.content)

topics = [
    "NULL values",
    "Duplicate Records",
    "Data Consistency"
]

dynamic_prompt = PromptTemplate.from_template("Explain {topic} from AI Data Engineer Perspective")

for topic in topics:
    ready_prompt = dynamic_prompt.invoke({ "topic" : topic})
    response = llm.invoke(ready_prompt)
    print(f"\nTopic : {topic}")
    print(response.content)"""

chat_template = ChatPromptTemplate.from_messages([
    ("system", "you are a {tone} Data Engineer Expert"),
    ("human", "Explain {topic} with practical examples")
])

ready_prompt = chat_template.invoke({
    "topic" : "Data Skew",
    "tone" : "Professional"
})
response = llm.invoke(ready_prompt)

print(response.content)