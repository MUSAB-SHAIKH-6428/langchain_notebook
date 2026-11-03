
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found")

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Send a request
response = llm.invoke("explain what NULL values are.")

# Display the response
print(response.content)


llm = ChatGroq(model="openai/gpt-oss-20b", temperature = 0.7)

res = llm.invoke("duplicate records.")

print(res.content)
