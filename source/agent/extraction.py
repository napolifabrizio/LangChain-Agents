from agent.environs import Environs
from pydantic import BaseModel, Field #Importação atualizada
from typing import List
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

text = '''A Apple foi fundada em 1 de abril de 1976 por Steve Wozniak, Steve Jobs e Ronald Wayne
com o nome de Apple Computers, na Califórnia. O nome foi escolhido por Jobs após a visita do pomar
de maçãs da fazenda de Robert Friedland, também pelo fato do nome soar bem e ficar antes da Atari
nas listas telefônicas.

O primeiro protótipo da empresa foi o Apple I que foi demonstrado na Homebrew Computer Club em 1975,
as vendas começaram em julho de 1976 com o preço de US$ 666,66, aproximadamente 200 unidades foram
vendidas,[21] em 1977 a empresa conseguiu o aporte de Mike Markkula e um empréstimo do Bank of America.'''

class Event(BaseModel):
    # '''Informação sobre um acontecimento'''
    date: str = Field(description='Data do acontecimento no formato YYYY-MM-DD')
    event: str = Field(description='Acontecimento extraído do texto')

class Events(BaseModel):
    # """Acontecimentos para extração"""
    events: List[Event] = Field(description='Lista de acontecimentos presentes no texto informado')

event_tool = convert_to_openai_function(Events)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Extraia as frases de acontecimentos. Elas devem ser extraídas integralmente'),
    ('user', '{input}')
])

chat = ChatOpenAI(model="gpt-4.1-mini")
chain = (
    prompt
    | chat.bind(functions=[event_tool], function_call={'name': 'Events'})
    | JsonOutputFunctionsParser()
)

res = chain.invoke({'input': text})
print(res)

