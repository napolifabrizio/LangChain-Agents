from agent.environs import Environs
from pydantic import BaseModel, Field
from langchain.agents import tool


class CurrentTemperatureArgs(BaseModel):
    local: str = Field(description='Localidade a ser buscada', examples=['São Paulo', 'Porto Alegre'])

@tool(args_schema=CurrentTemperatureArgs)
def current_temperature(local: str):
    '''Faz busca online de temperatura de uma localidade'''
    return '25ºC'

res = current_temperature.invoke({'local': 'Porto Alegre'})

print(res)