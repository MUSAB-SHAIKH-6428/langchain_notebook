from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

class FeedbackClassification(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Classify customer feedback as positive or negative.")

structured_llm = llm.with_structured_output(FeedbackClassification)

prompt = ChatPromptTemplate.from_messages([
    ("system","Act as a Customer Care"),
    ("human", "classify customer feedback : {issue}.")
])

positive_prompt = ChatPromptTemplate.from_messages([
    ("system","Act as a Customer Care"),
    ("human", "Generate a short appreciation response to the customer : {issue}.")
])

negative_prompt = ChatPromptTemplate.from_messages([
    ("system","Act as a Customer Care"),
    ("human", "Generate a professional apology and escalation response to the customer. : {issue}.")
])

positive_chain = positive_prompt | llm | StrOutputParser()

negative_chain = negative_prompt | llm | StrOutputParser()

def extract_sentiment(result)-> str:
    return result.sentiment

sentiment_chain = (
    prompt
    | structured_llm
    | RunnableLambda(extract_sentiment)
)

preserve_and_classify = RunnablePassthrough.assign(
    sentiment=sentiment_chain
)

conditional_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", positive_chain),
    negative_chain
)

final_orchestration = preserve_and_classify | conditional_chain

final_result = final_orchestration.invoke({
    "issue": "The payment was deducted twice, and customer support has ignored my complaints for three days."
})

print(final_result)