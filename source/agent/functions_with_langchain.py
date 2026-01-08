from agent.environs import Environs

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

environs = Environs()

## Criando a estrutura para mostrar á LLM, as funções que ela pode usar
class UnidadeEnum(str, Enum):
    celsius = 'celsius'
    fahrenheit = 'fahrenheit'

class ObterTemperaturaAtual(BaseModel):
    """Obtém a temperatura atual de uma determinada localidade"""
    local: str = Field(description='O nome da cidade', examples=['São Paulo', 'Porto Alegre'])
    unidade: Optional[UnidadeEnum]

from langchain_core.utils.function_calling import convert_to_openai_function

tool_temperatura = convert_to_openai_function(ObterTemperaturaAtual)

## Mostrando para a LLM, as funções que ela pode usar
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()
chat_tool_temperature = chat.bind(functions=[tool_temperatura])
# res = chat_tool_temperature.invoke('Qual é a temperatura de Porto Alegre')

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um assistente amigável chamado Isaac'),
    ('user', '{input}')
])

chain = prompt | chat_tool_temperature

res = chain.invoke({'input': "Olá! como está o tempo em Floripa?"})
print(res)

