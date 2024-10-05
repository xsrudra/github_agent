import os
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
import asyncio


llm = AzureChatOpenAI(
                        deployment_name="gpt-4o-mini",
                        openai_api_version="2024-02-15-preview",
                        model_name="gpt-4o-mini",
                        azure_endpoint="https://openai-eastus-instance-02.openai.azure.com/",
                        api_key=os.getenv("GPT_API_KEY")
                   )
# llm=ChatGroq(model="llama-3.2-3b-preview",api_key="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")
    