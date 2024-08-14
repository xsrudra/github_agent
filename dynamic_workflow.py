from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncIterator
import asyncio
import json
import base64
import os
from langgraph.graph import Graph, END
from langchain.agents import AgentExecutor, Tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_groq import ChatGroq
from requests_oauthlib import OAuth2Session
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.message import EmailMessage
import uvicorn
import pickle
from tools1 import hubspot_tool,google_sheets_tool,gmail_send_message
from langchain_openai import AzureChatOpenAI
load_dotenv



# class HubspotTool(BaseTool):
#     name = "Hubspot"
#     description = "Used for authenticating user with Hubspot and retrieving data"

#     def _run(self, query: str) -> str:
#         return hubspot_tool()

#     async def _arun(self, query: str) -> str:
#         return await asyncio.to_thread(self._run, query)

# class GoogleSheetsTool(BaseTool):
#     name = "Google_sheets"
#     description = "Used for entering data into Google Sheets"

#     def _run(self, query: str) -> str:
#         return google_sheets_tool()

#     async def _arun(self, query: str) -> str:
#         return await asyncio.to_thread(self._run, query)

# class SendMailTool(BaseTool):
#     name = "send_mail"
#     description = "Used for sending lead data via email"

#     def _run(self, to: str, subject: str, body: str) -> str:
#         return gmail_send_message(to, subject, body)

#     async def _arun(self, to: str, subject: str, body: str) -> str:
#         return await asyncio.to_thread(self._run, to, subject, body)


class HubspotTool(BaseTool):
    name = "Hubspot"
    description = "Used for authenticating user with Hubspot and retrieving data from hubspot"

    def _run(self, query: str) -> str:
        return hubspot_tool()

    async def _arun(self, query: str) -> str:
        return await asyncio.to_thread(self._run, query)

# class GoogleSheetsTool(BaseTool):
#     name = "Google_sheets"
#     description = "Used for entering multiple contacts data into single Google Sheet"

#     def _run(self, query: str) -> str:
#         return google_sheets_tool()

#     async def _arun(self, query: str) -> str:
#         return await asyncio.to_thread(self._run, query)

class GoogleSheetsTool(BaseTool):
    name = "Google_sheets"
    description = "Used for entering multiple contacts data into single Google Sheet"

    async def _arun(self, query: str = "") -> str:
        return await asyncio.to_thread(self._run, query)

    def _run(self, query: str = "") -> str:
        return google_sheets_tool()

class SendMailTool(BaseTool):
    name = "send_mail"
    description = "Used for sending lead data via  single email"

    def _run(self, to: str, subject: str, body: str) -> str:
        return gmail_send_message(to, subject, body)

    async def _arun(self, to: str, subject: str, body: str) -> str:
        return await asyncio.to_thread(self._run, to, subject, body)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BaseResponse(BaseModel):
    type: str

class AgentResponse(BaseResponse):
    type: str = "agent_response"
    content: str
    image: Optional[str] = None

class UserInterventionRequest(BaseResponse):
    type: str = "user_intervention"
    message: str

class WorkflowComplete(BaseResponse):
    type: str = "workflow_complete"

class ErrorResponse(BaseResponse):
    type: str = "error"
    message: str

def create_agent(tools: List[Tool], system_message: str):
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
    )
    llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    openai_api_version="2024-02-15-preview",
                    model_name="gpt-4o",
                    temperature=0.2,
                    azure_endpoint="https://openai-eastus-instance-02.openai.azure.com/",
                    api_key=os.getenv("APIKEY")
                )
    #llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key="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")
    agent = OpenAIFunctionsAgent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)




class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()

# workflow_agent = create_agent(
#     [HubspotTool(), GoogleSheetsTool(), SendMailTool()],

#     system_message = """
# You are an AI assistant managing a lead generation workflow. 
# Your task is to guide the process through the following steps:
# 1. Choose a platform (Hubspot or Salesforce) by asking user choice
# 2. Retrieve and summarize lead data
# 3. Enter all the data into Google Sheets
# 4. Send an email with the lead summary by asking user for email.

# You have access to the following tools:
# - Hubspot: retrieve_lead_data(platform)
# - Google Sheets: enter_data_into_sheets(data)
# - Send Mail: send_email(to, subject, body)

# For each step, determine if user input is required. If so, formulate a question for the user. and the question should include USER INPUT REQUIRED
# If no user input is needed, proceed with the next step of the workflow.
# """
#)

workflow_agent = create_agent(
    [HubspotTool(), GoogleSheetsTool(), SendMailTool()],
    system_message="""
    You are an AI assistant managing a lead generation workflow. 
    Your task is to guide the process through the following steps:
    1. Choose a platform (Hubspot or Salesforce) by asking user choice
    2. Retrieve and summarize lead data
    3. Enter all the data into Google Sheets
    4. Ask for an email address to send the lead summary
    5. Send an email with the lead summary
    6. Conclude the workflow
    You have access to the following tools:
    - Hubspot: retrieve_lead_data(platform)
    - Google Sheets: enter_data_into_sheets(data)
    - Send Mail: send_email(to, subject, body)

    For each step, determine if user input is required. If so, formulate a question for the user and include "USER INPUT REQUIRED:" before the question.
    If no user input is needed, proceed with the next step of the workflow.

    Important: 
    - When asking for an email address, use the exact phrase "USER INPUT REQUIRED: Please provide the email address".
    - After receiving an email address, proceed to send the email without asking for confirmation.
    - After sending the email, conclude the workflow with a summary of actions taken and include the phrase "WORKFLOW COMPLETE".
    - Use the provided context to determine the current step and avoid repeating completed steps.
    - Always include the relevant keyword (HUBSPOT:, SHEETS:, or EMAIL:) in your responses when performing actions with these tools.
    """
)





workflow_agent = create_agent(
    [HubspotTool(), GoogleSheetsTool(), SendMailTool()],
    system_message = """
    You are an AI assistant managing a lead generation workflow. 
    Your task is to guide the process through the following steps:
    1. Choose a platform (Hubspot or Salesforce) by asking user choice
    2. Retrieve and summarize lead data
    3. Enter all the data into Google Sheets
    4. Ask for an email address to send the lead summary
    5. Send an email with the lead summary
    6. Conclude the workflow

    You have access to the following tools:
    - Hubspot: retrieve_lead_data(platform)
    - Google Sheets: enter_data_into_sheets(data)
    - Send Mail: send_email(to, subject, body)

    For each step, determine if user input is required. If so, formulate a question for the user and include "USER INPUT REQUIRED:" before the question.
    If no user input is needed, proceed with the next step of the workflow.
    
    Important: 
    - When asking for an email address, use the exact phrase "USER INPUT REQUIRED: Please provide the email address".
    - After receiving an email address, proceed to send the email without asking for confirmation.
    - After sending the email, conclude the workflow with a summary of actions taken and include the phrase "WORKFLOW COMPLETE".
    """
)

async def workflow_step(state: Dict[str, Any], websocket: WebSocket) -> Dict[str, Any]:
    response = await workflow_agent.ainvoke({
        "input": state.get("input", ""),
        "chat_history": state.get("chat_history", [])
    })
    output = response["output"]
    
    if "USER INPUT REQUIRED:" in output:
        question = output.split("USER INPUT REQUIRED:")[1].strip()
        user_input = await get_user_input(websocket, question)
        state["input"] = user_input
        state["chat_history"] = state.get("chat_history", []) + [HumanMessage(content=user_input)]
        
        if "Please provide the email address" in question:
            state["email_address"] = user_input
        
        return state
    
    if "HUBSPOT:" in output:
        image = load_image("hubspot.png")
    elif "GOOGLE SHEETS:" in output:
        image = load_image("sheets.png")
    elif "EMAIL:" in output:
        image = load_image("gmail.png")
    else:
        image = None
    
    await manager.send_personal_message(
        AgentResponse(content=output, image=image).dict(),
        websocket
    )
    
    state["chat_history"] = state.get("chat_history", []) + [SystemMessage(content=output)]
    state["input"] = ""
    
    if "WORKFLOW COMPLETE" in output:
        return END
    
    return state

async def get_user_input(websocket: WebSocket, prompt: str) -> str:
    await manager.send_personal_message(
        UserInterventionRequest(message=prompt).dict(),
        websocket
    )
    response = await websocket.receive_text()
    try:
        data = json.loads(response)
        if data["type"] == "user_intervention":
            return data["message"]
    except (ValueError, KeyError):
        return response
    return ""

def load_image(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'images', filename)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

workflow = Graph()
workflow.add_node("workflow_step", workflow_step)
workflow.set_entry_point("workflow_step")
workflow.add_edge("workflow_step", "workflow_step")



@app.websocket("/ws/lead_generation")
async def websocket_lead_generation(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        user_input = await websocket.receive_text()
        state = {"input": user_input, "chat_history": [], "websocket": websocket}
        
        while True:
            new_state = await workflow_step(state, websocket)
            if new_state == END:
                await manager.send_personal_message(WorkflowComplete().dict(), websocket)
                break
            state.update(new_state)
        
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_personal_message(
            ErrorResponse(message=f"An error occurred: {str(e)}").dict(),
            websocket
        )
    finally:
        manager.disconnect(websocket)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)