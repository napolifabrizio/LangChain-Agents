from langchain_openai import ChatOpenAI

from agent.environs import Environs

chat = ChatOpenAI()

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)

agent = create_pandas_dataframe_agent(
    chat,
    df,
    verbose=True,
    agent_type='tool-calling',
    allow_dangerous_code=True
)

inputs = [
    "Quantas linhas tem na tabela?",
    "Qual a média da idade dos passageiros?",
    "Quantos passageiros sobreviveram?"
]

# for inp in inputs:
#     res = agent.invoke({"input": inp})
#     print(inp)
#     print(res.get("output"))

# Usando por SQL

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

db = SQLDatabase.from_uri('sqlite:///scripts/arquivos/Chinook.db')

agent_executor = create_sql_agent(
    chat,
    db=db,
    agent_type='tool-calling',
    verbose=True
)

res = agent_executor.invoke({'input': 'Me descreva a base de dados'})

print(res.get("output"))