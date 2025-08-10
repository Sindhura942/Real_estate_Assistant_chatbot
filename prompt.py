from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template

updated_template = "You are a helpful assistant for RealEstate research." + template
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "Use the given context to answer the question. "
     "If you don't know the answer, say you don't know. "
     "Limit your answer to three sentences.\n\nContext:\n{context}"),
    ("human", "{input}")
])


EXAMPLE_PROMPT = ChatPromptTemplate.from_template(
    "Content: {page_content}\nSource: {source}"
)
