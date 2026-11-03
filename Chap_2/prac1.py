import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("human", "Explain {topic} with a practical example. also set the level as {difficulty}"),
    ("system","You are a data Engineer Expert")
    ])

str_parser = StrOutputParser()

def dictionary_maker(text : str) -> dict:
    text = text.strip().lower()
    return {"text": text}
dictionary_maker_runnable = RunnableLambda(dictionary_maker)

chain = prompt_template | llm | str_parser | dictionary_maker_runnable

result = chain.invoke({
    "topic" : "Data NULLs",
    "difficulty" : "Easy"
})

print(result)

#task 1
summary_prompt_template = ChatPromptTemplate.from_messages([
    ("human", "Generate a short summary sentence for issue : {issue}."),
    ("system","Act as a Customer Care")
    ])
#task 2
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
#task 3
str_parser = StrOutputParser()

summary_chain = summary_prompt_template | llm | str_parser

def classification_chain(text : dict):
    text = text["issue"]

    complaint_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Business Analyst"),
    ("human", "Categorize this complaint for analytics : {complaint}")])

    # Just for understanding
    """llm = ChatGroq(model="openai/gpt-oss-20b", temperature = 0)

    str_parser = StrOutputParser()"""

    chain_classification = complaint_prompt | llm | str_parser

    result = chain_classification.invoke({"complaint": text})

    return result

runnable_classification_chain = RunnableLambda(classification_chain)


parallel_chain = RunnableParallel(
    summary = summary_chain,
    classification = runnable_classification_chain
)

result = parallel_chain.invoke({
    "issue": "Customer 1042 was charged twice for the monthly subscription."
})

print(result)