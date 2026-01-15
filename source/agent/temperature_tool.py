import requests
import datetime

from langchain.agents import tool
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function

from agent.environs import Environs


class CurrentTemperatureArgs(BaseModel):
    latitude: float = Field(description='Latitude da localidade que buscamos a temperatura')
    longitude: float = Field(description='Longitude da localidade que buscamos a temperatura')



chat = ChatOpenAI()

@tool(args_schema=CurrentTemperatureArgs)
def current_temperature(latitude: float, longitude: float):
    '''Retorna a temperatura atual para uma dada coordenada'''

    URL = 'https://api.open-meteo.com/v1/forecast'

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    response = requests.get(URL, params=params)
    if response.status_code == 200:
        result = response.json()

        date_now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        hours = [datetime.datetime.fromisoformat(temp_str) for temp_str in result['hourly']['time']]
        closest_index = min(range(len(hours)), key=lambda x: abs(hours[x] - date_now))

        temp_atual = result['hourly']['temperature_2m'][closest_index]
        return temp_atual
    else:
        raise Exception(f'Request para API {URL} falhou: {response.status_code}')

tools = [current_temperature]
json_tools = [convert_to_openai_function(tool) for tool in tools]
tool_run = {tool.name: tool for tool in tools}

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um assistente amigável chamado Isaac'),
    ('user', '{input}')
])

def routing(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        return tool_run[result.tool].run(result.tool_input)

chain = (
    prompt
    | chat.bind(functions=json_tools)
    | OpenAIFunctionsAgentOutputParser()
    | routing
)

res = chain.invoke({"input": "Qual é a temperatura atual de Anil no Brasil em Rio de Janeiro?"})
# print(res)
