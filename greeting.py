from langchain_openai import AzureChatOpenAI
import os
from model import llm
import os
import asyncio
from langchain_groq import ChatGroq
from functools import lru_cache
import pandas as pd
#llm = ChatGroq(model="llama-3.1-70b-versatile", api_key="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")



async def greet(query: str,df:pd.DataFrame) -> str:
    greeting_needed = await asyncio.to_thread(llm.invoke, (f"""Based on the query and dataframe {df}, determine if it is a greeting or a current affairs question.
                        Answer with 'Hello! I am your data analysis agent. Please ask a query relevant to data.' 
                        if the response to the query should be a greeting or current affairs question, 
                        or 'NO' if no greeting or relevant response is required and query is releavnt to data

                        Examples:
                        Query: 'Hello, how are you?'
                        Response: 'Hello! I am your data analysis agent. Please ask a query relevant to data.'

                        Query: 'What is the weather today?'
                        Response: 'Hello! I am your data analysis agent. Please ask a query relevant to data.'

                        Query: 'Who is the Prime Minister?'
                        Response: 'Hello! I am your data analysis agent. Please ask a query relevant to data.'

                        Query: 'What is the total sales revenue for last year?'
                        Response: NO

                        Query: 'List the top 5 selling products'
                        Response: NO

                        Query: 'What do you think about sales strategies?'
                        Response: NO

                        Now, based on the following query, determine if a greeting or current affairs question is needed:

                        Query: '{query}'
                        Response:
                        """))
    return greeting_needed.content






