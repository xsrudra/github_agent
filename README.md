import os
import asyncio
from model import llm
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
import asyncio



def charts(query):
        chart_needed = llm.invoke(f"""Based on the query, determine if a new chart should be generated, if an existing image should be referenced, or if no chart is required.
                        Answer only with 'YES' if the response to the query should be a visual chart, or 'NO' if no chart is required.

                        Examples:
                        Query: 'Show the monthly sales data for the last year'
                        Response: YES

                        Query: 'What is the total sales revenue for last year?'
                        Response: NO

                        Query: 'List the top 5 selling products'
                        Response: NO

                        Query: 'Show the trend of website visitors over the past 6 months'
                        Response: YES

                        Query: 'Provide the average monthly sales for this year'
                        Response: NO

                        Now, based on the following query, determine if a chart is needed:

                        Query: '{query}'
                        Response:
                        """)
        chart_needed_content = chart_needed.content.decode('utf-8') if isinstance(chart_needed.content, bytes) else chart_needed.content   
        return chart_needed_content          
