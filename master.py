from fastapi import FastAPI, File, UploadFile,  status, Response,Query,Request,HTTPException,Body,Form,Depends,status
import uuid
import asyncio
import shutil
import yaml
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pandasai.connectors import PostgreSQLConnector, MySQLConnector, SQLConnector
from pandasai.ee.connectors import SnowFlakeConnector
from langchain_openai import AzureChatOpenAI
import pandasai
from pandasai import Agent
from typing import List, Optional, Dict
import os
from uuid import uuid4
from functools import lru_cache
import threading
from langchain_core.prompts import MessagesPlaceholder
from pandasai import SmartDataframe,Agent
from langchain_core.prompts import ChatPromptTemplate
from pandasai.connectors.yahoo_finance import YahooFinanceConnector
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
import io
from typing import Any
from dotenv import load_dotenv
from pandasai.ee.agents.judge_agent import JudgeAgent
from guardrails import Guard
#from rails.unusual_prompt import UnusualPrompt
judge = JudgeAgent()
from langchain.memory import ConversationBufferMemory
import litellm
import base64
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
import json
from psycopg2 import sql
from sqlalchemy import Date, Float, Integer, Numeric, String, create_engine, inspect,text
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
from functools import wraps
from greeting import greet
from chart import charts
from prompts import get_prompts
from model import llm
from converse import converse
import cProfile
import pstats
import io
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import asyncio
from collections import defaultdict
import json
from cachetools import TTLCache
from langchain.tools  import Tool
from langchain.agents import initialize_agent, AgentType
asyncio.get_event_loop().set_debug(True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Database(BaseModel):
    host: Optional[str] = None
    port: Optional[str] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    tables: Optional[List[str]] = None
    databasetype: Optional[str] = None
    account: Optional[str] = None
    warehouse: Optional[str] = None
    dbSchema: Optional[str] = None
    driver: Optional[str] = None
    action:str=None

os.environ['PANDASAI_API_KEY'] ="$2a$10$T0.tWMY89DwvAo0eda0dZuOoKdArKwuOEDDeMcDrP/oiM/hsi6cX."  
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
cache = TTLCache(maxsize=100, ttl=300)  # Cache up to 100 items for 5 minutes

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)
class AgentManager:
    def __init__(self):
        self.agent = None
    
    def initialize_agent(self, sdf, llm1, description, judge):
        if self.agent is None:  
            self.agent = Agent(
                sdf, 
                memory_size=15, 
                config={
                    "llm": llm1,
                    "data_viz_library": "seaborn",
                    "open_charts": False,
                    "enable_cache": True,
                    "max_retries": 4,
                    "description": description
                },
                judge=judge
            )
        return self.agent    
    def clear_agents(self):
        self.agent = None

@lru_cache()  
def get_agent_manager():
    return AgentManager()  

# Profiling decorator
# def profile(func):
#     @wraps(func)
#     async def wrapper(*args, **kwargs):
#         pr = cProfile.Profile()
#         pr.enable()
#         result = await func(*args, **kwargs)
#         pr.disable()
#         s = io.StringIO()
#         ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
#         ps.print_stats()
#         print(s.getvalue())
#         return result
#     return wrapper  
        
#########################################################################################################################################

#POST API TO UPLOAD FILE
@app.post("/upload-file/")

async def upload_file(file: UploadFile = File(...)):
   
    try:
        contents = await file.read()
        unique_id = str(uuid.uuid4())
        folder_path = os.path.join('data', 'csv', unique_id)   
        os.makedirs(folder_path, exist_ok=True)

        if file.filename.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, f"{unique_id}.csv")
            with open(csv_file_path, 'wb') as f:
                f.write(contents)

        elif file.filename.endswith(('.xls', '.xlsx')):
            excel_file_path = os.path.join(folder_path, file.filename)
            with open(excel_file_path, 'wb') as f:
                f.write(contents)
        else:
            return {"message": "Unsupported file format", "error": True}

        return {"message": f"File '{file.filename}' uploaded successfully to folder'","uuid":unique_id, "error": False}
    
    except Exception as e:
        return {"message": "Error processing file", "error": True, "exp": str(e)}
    

#########################################################################################################################################################

#POST API TO CONNECT TO DATABASES
@app.post("/database-connection/")

async def connect_to_database(response: Response, db: Database):
    if db.databasetype=='postgres':

        try:
            unique_id = str(uuid4())
            dir_path = f"data/postgres/{unique_id}"
            file_path = f"{dir_path}/{unique_id}.yaml"
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "w") as file:
                yaml.dump(db.dict(), file)

            response.status_code = status.HTTP_200_OK
            return {"message": "Database credentials saved", "uuid": unique_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
#############################################################################################################################################
    
####################################################################################################################################################################      
#POST API TO QUERY DATA
@app.post("/query")
async def process_query(
    response: Response,
    id: str = Query(..., description="UUID string"),
    query: str = Body(..., description="Query text"),
    type: str = Body(..., description="database type"),
    agent_manager: AgentManager = Depends(get_agent_manager)) -> Dict[str, Any]:
    cache_key = f"{id}:{query}:{type}"
    if cache_key in cache:
        return cache[cache_key]
    
    
    
    
    tools = [
       Tool(
           name="Fault_Query_Agent",
           func=converse,
           description="used for answering queries related to troubleshooting of faults and relation between different components."
       ),
       Tool(
           name="Plot_Agent",
           func=agent_manager.agent.chat,
           description="This is used for plotting visualizations, answering queries related to historical faults."
       ),
       Tool(
           name="RCA_Query_Agent",
           func=root_cause_analysis_query_agent.chat,
           description="Use for Root Cause Analysis (RCA) Questions."
       )
   ]


    agent = initialize_agent(
       tools=tools,
       llm=llm,
       verbose=True,
       handle_parsing_error=True,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
   )
    try:
        if type == 'postgres':
            result = await process_postgres_query(response, id, query, agent_manager)
        elif type == 'file':
            result = await process_file_query(response, id, query, agent_manager)
        else:
            raise ValueError("Invalid data source type")

        cache[cache_key] = result
        return result
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"message": "Error processing query", "error": True, "exp": str(e)}

async def process_postgres_query(response: Response, id: str, query: str, agent_manager: AgentManager) -> Dict[str, Any]:
    agent = agent_manager.agent
    memory.load_memory_variables({})
    
    chart_needed =  charts(query)
    
    
    if chart_needed.strip().lower() == 'yes':
        result = await asyncio.to_thread(agent.chat, query)
        text, data = process_result(result)
    else:
        result = await process_sql_query(id, query)
        text = result.get('output', '')
        data = pd.DataFrame()

    image_base64,text =  process_image(text)
    json_data = data.to_json(orient='records') if not data.empty else ""

    
    df = pd.read_csv(f"data/postgres/{id}/{id}.csv")
    ques_df = await get_prompts(df.head(), id, 'postgres')
    
    response.status_code = status.HTTP_200_OK
    return {
        "base64_image": image_base64,
        "text": str(text),
       "dataframe": json_data,
        "questions": ques_df,
        "error": False,
    }



async def process_file_query(response: Response, id: str, query: str, agent_manager: AgentManager) -> Dict[str, Any]:
    if not id:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "No data source connected. Please connect to a database or upload a file first.", "error": True}

    path = f"data/csv/{id}/{id}.csv"
    df = pd.read_csv(path)


    greet_task = greet(query,df.head())
    prompts_task = get_prompts(df.head(), id, 'file')
    greeting, ques_df = await asyncio.gather(greet_task, prompts_task)


    print('After greet')
    if greeting.strip().lower() != 'no':
        text = greeting
        data =pd.DataFrame()
    else:
        agent = agent_manager.agent
        result = await asyncio.to_thread(agent.chat, query)
        text, data = process_result(result)

    image_base64, text = process_image(text)
    json_data = data.to_json(orient='records') if not data.empty else ""


    response.status_code = status.HTTP_200_OK
    return {
        "base64_image": image_base64,
        "text": str(text),
        "dataframe": json_data,
        "questions": ques_df,
        "error": False,
    }


async def process_sql_query(id: str, query: str) -> Dict[str, Any]:
    system = """You are an agent designed to interact with databases. Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Previous conversation history:
    {chat_history}
    QUERY FOR ROWS AND COLUMNS THAT ARE PRESENT IN THE TABLE. You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. The input may have names in lower case. If querying, make your query align by using the LIKE operator in the query such that they have the most similarity.
    THERE MAY BE WHITESPACES IN BETWEEN; CONSIDER THIS POSSIBILITY AS WELL. FOR RESULTS HAVING COST, DISPLAY DOLLAR or RUPEE SIGN IN THE RESULT.
    DONT DISPLAY '*' IN THE RESPONSE. DONT SHOW ANY SAMPLE REPRESENTATIONS."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    file_path = f"data/postgres/{id}/{id}.yaml"
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        db_data = Database(**data)

    conn_str = f"postgresql://{db_data.username}:{db_data.password}@{db_data.host}:{db_data.port}/{db_data.database}"
    db = SQLDatabase.from_uri(conn_str, include_tables=db_data.tables)

    sql_agent = create_sql_agent(
        llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prompt=prompt,
        handle_parsing_errors=True,
        agent_executor_kwargs={
            "memory": memory,
            "return_intermediate_steps": True
        }
    )

    result = await asyncio.to_thread(sql_agent.invoke, {"input": query})
    memory.save_context({"input": query}, {"output": str(result.get('output', ''))})
    return result

def process_result(result):
    if isinstance(result, pd.DataFrame):
        return "Your response has been successfully generated. Kindly refer to the table for details.", result
    else:
        return str(result), pd.DataFrame()

def process_image(text: str) -> str:
    if isinstance(text, str) and ".png" in text:
        image_path = text
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        text =  converse(text)
        memory.save_context({"input": "image_generated"}, {"output": text})
        print(f"Image generated: {image_path}")
        return image_base64,text
    else:
        return "",text
   

                

    
##################################################################################################################################################
 
 #GET API TO GET DASHBOARD DATA
@app.get("/dashboard-data")

async def dashboard_data(
    response: Response,
    id: str = Query(..., description="UUID string"),
    type: str = Query(..., description="database type"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    try:
        results = []
        json_file_path = './exports/charts/temp_chart.json'
        with open(json_file_path, 'r') as file:
            image_data = json.load(file)

        if type == "postgres":
            results = await handle_postgres(id, agent_manager, image_data)
        elif type == "file":
            results = await handle_file(id, agent_manager, image_data)
        else:
            raise ValueError("Invalid data source type")

        response.status_code = status.HTTP_200_OK
        return {"data": results}

    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

async def handle_postgres(id: str, agent_manager: AgentManager, image_data: Dict) -> List[Dict]:
    file_path = f"data/postgres/{id}/{id}.yaml"
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        db_data = Database(**data)

    conn_str = f"postgresql://{db_data.username}:{db_data.password}@{db_data.host}:{db_data.port}/{db_data.database}"
    engine = create_engine(conn_str)
    
    try:
        with engine.connect() as connection:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            if not tables:
                raise ValueError("No tables found in the database")

            table_name = db_data.tables[0] if db_data.tables else tables[0]
            table_info = await get_table_info(connection, inspector, table_name)
            
            preview_data = await get_preview_data(connection, table_name)
            df = pd.DataFrame(preview_data)
            
            await save_csv(df, id, "postgres")
            
            questions = await get_prompts(df.head(), id, "postgres")
            
            connectors = [
                PostgreSQLConnector(config={
                    "host": db_data.host,
                    "port": int(db_data.port),
                    "database": db_data.database,
                    "username": db_data.username,
                    "password": db_data.password,
                    "table": table,
                }) for table in db_data.tables
            ]
            
            description = "You are a PostgreSQL database analysis agent. Your main goal is to help non-technical users analyze data from a PostgreSQL database. Formulate SQL queries when needed and provide clear explanations of the results."
            agent_manager.initialize_agent(connectors, llm, description, judge)
            
            return [{**table_info, 'questions': questions, 'image': image_data,"data_preview":preview_data}]
    finally:
        engine.dispose()

async def handle_file(id: str, agent_manager: AgentManager, image_data: Dict) -> List[Dict]:
    path = f"data/csv/{id}/{id}.csv"
    df = pd.read_csv(path)
    sdf = SmartDataframe(path)

    description = "You are an Excel/CSV data analysis agent. Your main goal is to help non-technical users analyze data from CSV or Excel sheets. Use pandas operations to process and analyze the data effectively. For visualizations, generate interactive Plotly figures and provide them in JSON format. This enables users to render and interact with charts directly within their web applications. In case of greetings, respond with a greeting. Be sure to understand the user query and then proceed to answer the questions. Use different colors when plotting the charts. DON'T OPEN THE CHART"

    agent_manager.initialize_agent(sdf, llm, description, judge)

    numeric_summaries = {
        'total_records': len(df),
        'number_of_numeric_columns': len(df.select_dtypes(include='number').columns),
        'number_of_non_numeric_columns': len(df.select_dtypes(exclude='number').columns)
    }

    preview_rows = df.head(15).to_dict(orient='records')
    questions = await get_prompts(df.head(), id, "file")

    return [{
        'numeric_summaries': numeric_summaries,
        'data_preview': preview_rows,
        'image': image_data,
        'questions': questions
    }]

async def get_table_info(connection, inspector, table_name: str) -> Dict:
    total_records = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    columns_info = inspector.get_columns(table_name)
    
    numeric_types = (Integer, Float, Numeric)
    non_numeric_types = (String, Date)
    
    numeric_columns = [col['name'] for col in columns_info if isinstance(col['type'], numeric_types)]
    non_numeric_columns = [col['name'] for col in columns_info if isinstance(col['type'], non_numeric_types)]
    
    return {
        'numeric_summaries': {
            'total_records': total_records,
            'number_of_numeric_columns': len(numeric_columns),
            'number_of_non_numeric_columns': len(non_numeric_columns)
        }
    }

async def get_preview_data(connection, table_name: str) -> List[Dict]:
    query = text(f"SELECT * FROM {table_name} LIMIT 15")
    result = connection.execute(query)
    column_names = result.keys()
    return [dict(zip(column_names, row)) for row in result.fetchall()]

async def save_csv(df: pd.DataFrame, id: str, type: str):
    directory = f'data/{type}/{id}'
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{id}.csv")
    df.to_csv(path, index=False)


###########################################################################################################################################

@app.post("/end-chat/")

async def end_chat(response: Response, type: str = Body(..., description="database type"), id: str = Query(..., description="UUID string"),agent_manager: AgentManager = Depends(get_agent_manager)):
    if type=="postgres":

        pandasai.clear_cache()
        folder_path = f"data/postgres/{id}"
        file_path = os.path.join(folder_path, f"{id}.yaml")
        agent_manager.clear_agents()
        
        memory.clear()
        if os.path.exists(file_path):
            shutil.rmtree(folder_path)  
            response.status_code = status.HTTP_200_OK
            return {"message": "Chat ended successfully"}
        else:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"error": "Please upload the file"}  
    
    elif type == "file":
        pandasai.clear_cache()
       
        folder_path = f"data/csv/{id}"
        csv_file_path = os.path.join(folder_path, f"{id}.csv")
        agent_manager.clear_agents()
        memory.clear()
        if os.path.exists(csv_file_path):
            shutil.rmtree(folder_path)  
            response.status_code = status.HTTP_200_OK
            return {"message": "Chat ended successfully"}
        else:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"error": "Please upload the file"}
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Please connect to a data source to proceed with the analysis"}

    
@app.post("/get-table-names")
# @profile
async def table_names(connection: Database, response: Response):    
    
    if connection.databasetype == "postgres":
        print("Establishing PostgreSQL connection")
        conn = f"postgresql+psycopg2://{connection.username}:{connection.password}@{connection.host}:{connection.port}/{connection.database}"
        engine = create_engine(conn)        
        try:
            inspector = inspect(engine)
            tables = inspector.get_table_names()            
            print("\nTables in the database:")
            for table in tables:
                print(table)            
            if not tables:
                response.status_code = status.HTTP_404_NOT_FOUND
                return {"error": "No tables found in the database"}            
            return tables
        except SQLAlchemyError as e:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error": f"Database error: {str(e)}"}        
        finally:
            # Close the engine
            engine.dispose()
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Unsupported database type"}

@app.get("/")
async def server():
    return {" message: Server started"}
