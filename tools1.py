from __future__ import print_function
import mimetypes
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
from langchain_openai import AzureChatOpenAI

import asyncio
from typing import AsyncIterator, Dict, List, Any, Optional
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.graph import Graph, END
from langchain.agents import AgentExecutor, Tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_groq import ChatGroq
from langchain.prompts import  MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import base64
from fastapi.websockets import WebSocketDisconnect
from fastapi import FastAPI, HTTPException, Response, WebSocket,Request
from fastapi.middleware.cors import CORSMiddleware
from requests_oauthlib import OAuth2Session  
import os
import pickle
import json
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import logging
import pickle
import base64
import os.path
import json
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.errors import HttpError
from langchain_openai import AzureChatOpenAI
import base64
from email.message import EmailMessage
import google.auth
from googleapiclient.discovery import build

from requests_oauthlib import OAuth2Session
from gsheets import Sheets
import pickle
import json
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.message import EmailMessage
import uvicorn


ID=''

#####################################################################################################################
CLIENT_ID     = '0af9998a-56ab-4d56-8927-fe759d189952'
CLIENT_SECRET = '30530718-0286-41ea-9de6-a0e27e833122'

SCOPES        = ['crm.objects.contacts.read','oauth','crm.objects.leads.read']



def hubspot_tool():
    """
    Connects to HubSpot, fetches all contacts, and then filters for leads.
    """
    app_config = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scopes': SCOPES,
        'auth_uri': 'https://app.hubspot.com/oauth/authorize',
        'token_uri': 'https://api.hubapi.com/oauth/v1/token'
    }

    if os.path.exists('hstoken.pickle'):
        with open('hstoken.pickle', 'rb') as tokenfile:
            token = pickle.load(tokenfile)
    else:
        token = InstallAppAndCreateToken(app_config)
        SaveTokenToFile(token)

    hubspot = OAuth2Session(
        app_config['client_id'],
        token=token,
        auto_refresh_url=app_config['token_uri'],
        auto_refresh_kwargs=app_config,
        token_updater=SaveTokenToFile
    )

    # Define the properties we want to retrieve
    properties = [
        "firstname", "lastname", "email", "company", "phone", "website",
        "hs_lead_status", "lifecyclestage", "hs_analytics_source", "industry"
    ]

    
    url = 'https://api.hubapi.com/crm/v3/objects/contacts'
    params = {
        'properties': ','.join(properties),
        'limit': 100  # Adjust as needed
    }
    response = hubspot.get(url, params=params)
    response_data = response.json()

    # Save the full response to a file
    with open('all_contacts_data.json', 'w') as file:
        json.dump(response_data, file, indent=4)

    leads = []
    other_contacts = []

    if 'results' in response_data:
        for contact in response_data['results']:
            contact_id = contact['id']
            contact_properties = contact['properties']
            
            # Print all properties for debugging
            print(f"Contact ID: {contact_id}")
            print(f"Contact Properties:")
            for prop, value in contact_properties.items():
                print(f"  {prop}: {value}")
            print("---")

            # Check if the contact is a lead
            if contact_properties.get('hs_lead_status') == 'None':
                other_contacts.append(contact)
            else:
                
                leads.append(contact)

    print(f"\nTotal contacts retrieved: {len(response_data.get('results', []))}")
    print(f"Leads found: {len(leads)}")
    print(f"Other contacts: {len(other_contacts)}")

    # Save leads to a separate file
    with open('leads_data.json', 'w') as file:
        json.dump(leads, file, indent=4)

    return leads
def InstallAppAndCreateToken(config, port=8088):
    """
    Creates a simple local web app+server to authorize your app with a HubSpot hub.
    Returns the refresh and access token.
    """  
    from wsgiref import simple_server
    import webbrowser
    local_webapp = SimpleAuthCallbackApp()
    local_webserver = simple_server.make_server(host='localhost', port=port, app=local_webapp)
    redirect_uri = 'http://{}:{}/'.format('localhost', local_webserver.server_port)
    oauth = OAuth2Session(
        client_id=config['client_id'],
        scope=config['scopes'],
        redirect_uri=redirect_uri
    )
    auth_url, _ = oauth.authorization_url(config['auth_uri'])    
    print('-- Authorizing your app via Browser --')
    print('If your browser does not open automatically, visit this URL:')
    print(auth_url)
    webbrowser.open(auth_url, new=1, autoraise=True)
    local_webserver.handle_request() 
    auth_response = local_webapp.request_uri.replace('http','https')
    token = oauth.fetch_token(
        config['token_uri'],
        authorization_response=auth_response,
      
        include_client_id=True,
        client_secret=config['client_secret']
    )
    return token

class SimpleAuthCallbackApp(object):
    """
    Used by our simple server to receive and 
    save the callback data authorization.
    """
    def __init__(self):
        self.request_uri = None
        self._success_message = (
            'All set! Your app is authorized.  ' + 
            'You can close this window now and go back where you started from.'
        )

    def __call__(self, environ, start_response):
        from wsgiref.util import request_uri
        
        start_response('200 OK', [('Content-type', 'text/plain')])
        self.request_uri = request_uri(environ)
        return [self._success_message.encode('utf-8')]

def SaveTokenToFile(token):
    """
    Saves the current token to file for use in future sessions.
    """
    with open('hstoken.pickle', 'wb') as tokenfile:
        pickle.dump(token, tokenfile)
        
###############################################################################################################################################

        
########################################################################################################################



def google_sheets_tool():
    """
    Integrates lead data with Google Sheets.
    """
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://mail.google.com/"]
   
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client-secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
            
    service = build("sheets", "v4", credentials=creds)

    try:
        spreadsheet = {"properties": {"title": "HubSpot Leads"}}
        spreadsheet = service.spreadsheets().create(body=spreadsheet, fields="spreadsheetId").execute()
        spreadsheet_id = spreadsheet.get('spreadsheetId')
        print(f"Spreadsheet ID: {spreadsheet_id}")

        # Prepare headers and data
        headers = ["ID", "Created At", "Updated At", "First Name", "Last Name", "Email", "Company", "Phone", "Website", "Lead Status", "Lifecycle Stage", "Source", "Industry"]
        values = [headers]  # Start with headers

        with open("leads_data.json", 'r') as file:
            lead_data = json.load(file)

        for lead in lead_data:
            properties = lead['properties']
            row = [
                lead['id'],
                lead['createdAt'],
                lead['updatedAt'],
                properties.get('firstname', ''),
                properties.get('lastname', ''),
                properties.get('email', ''),
                properties.get('company', ''),
                properties.get('phone', ''),
                properties.get('website', ''),
                properties.get('hs_lead_status', ''),
                properties.get('lifecyclestage', ''),
                properties.get('hs_analytics_source', ''),
                properties.get('industry', '')
            ]
            values.append(row)

        body = {"values": values}
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A1',  
            valueInputOption='RAW',
            body=body
        ).execute()

        print(f"{result.get('updatedCells')} cells updated.")

        # Export to CSV
        sheets = Sheets.from_files('client-secret.json', 'storage.json')
        url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}'
        s = sheets.get(url)
        s.sheets[0].to_csv('hubspot_leads.csv', encoding='utf-8', dialect='excel')

        return spreadsheet_id

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def gmail_send_message(to: str, subject: str, body: str):
    """Create and send an email message"""
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly","https://www.googleapis.com/auth/spreadsheets","https://mail.google.com/"]
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client-secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        service = build("gmail", "v1", credentials=creds)
        message = EmailMessage()

        message.set_content(body)

        message["To"] = to
        message["From"] = "testingops2@gmail.com"
        message["Subject"] = subject
        
        ctype, encoding = mimetypes.guess_type("hubspot_leads.csv")
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        with open("hubspot_leads.csv", "rb") as fp:
            attachment_data = fp.read()
        message.add_attachment(attachment_data, maintype=maintype, subtype=subtype, filename=os.path.basename("hubspot_leads.csv"))

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
    
        send_message = (
            service.users()
            .messages()
            .send(userId="me", body=create_message)
            .execute()
        )
        print(f'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(f"An error occurred: {error}")
        send_message = None
    return f"Message sent to {to} with ID {send_message['id']}"

def platform_tool():
    return "Used for asking the choice of platform user wants"