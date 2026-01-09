from agent.environs import Environs
from pydantic import BaseModel, Field #Importação atualizada
from typing import List
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


revenue = """Em um liquidificador, adicione a cenoura, os ovos e o óleo, depois misture.
Acrescente o açúcar e bata novamente por 5 minutos.
Em uma tigela ou na batedeira, adicione a farinha de trigo e depois misture novamente.
Acrescente o fermento e misture lentamente com uma colher.
Asse em um forno preaquecido a 180° C por aproximadamente 40 minutos.
Cobertura
Despeje em uma tigela a manteiga, o chocolate em pó, o açúcar e o leite, depois misture.
Leve a mistura ao fogo e continue misturando até obter uma consistência cremosa, depois despeje a calda por cima do bolo.
"""

class Ingredient(BaseModel):
    """
    Ingrediente para a receita
    """
    name: str = Field(description="O nome do ingrediente")

class Utensil(BaseModel):
    """
    Utensílio de cozinha para a receita
    """
    name: str = Field(description="O nome do utensílio de cozinha")

class RevenueMap(BaseModel):
    """
    Classe que lista os ingredientes e os utensílios de cozinha que
    são necessários para a receita dada pelo usuário
    """
    ingredients: List[Ingredient] = Field(description="Lista com os ingredientes")
    utensils: List[Ingredient] = Field(description="Lista com os utensílios")

revenue_tool = convert_to_openai_function(RevenueMap)

chat = ChatOpenAI()
chat = chat.bind(functions=[revenue_tool], function_call={'name': 'RevenueMap'})

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um assistente de cozinha, o usuário irá te passar uma receita, e você vai extrair os ingredientes e os utensílios de cozinha necessários para fazer a receita.'),
    ('user', '{input}')
])

chain = prompt | chat | JsonOutputFunctionsParser()

res = chain.invoke({"input": revenue})
print(res)