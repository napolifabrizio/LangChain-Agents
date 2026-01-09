from agent.environs import Environs

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

environs = Environs()

# class Feeling(BaseModel):
#     '''Define o sentimento e a língua da mensagem enviada'''
#     feeling: str = Field(description='Sentimendo do texto. Deve ser "pos", "neg" ou "nd" para não definido.')
#     language: str = Field(description='Língua que o texto foi escrito (deve estar no formato ISO 639-1)')

# tool_feeling = convert_to_openai_function(Feeling)

# prompt = ChatPromptTemplate.from_messages([
#     ('system', 'Pense com cuidado ao categorizar o texto conforme as instruções'),
#     ('user', '{input}')
# ])

# chat = ChatOpenAI()
# chat = chat.bind(functions=[tool_feeling], function_call={'name': 'Feeling'})

# chain = prompt | chat | JsonOutputFunctionsParser()

# res = chain.invoke({"input": "Eu gosto muito de hamburguer"})

# print(res)

## Exemplo

doubts = [
    'Bom dia, gostaria de saber se há um certificado final para cada trilha ou se os certificados são somente para os cursos e projetos? Obrigado!',
    'In Etsy, Amazon, eBay, Shopify https://pint77.com Pinterest+SEO +II = high sales results',
    'Boa tarde, estou iniciando hoje e estou perdido. Tenho vários objetivos. Não sei nada programação, exceto que utilizo o Power automate desktop da Microsoft. Quero aprender tudo na plataforma que se relacione ao Trading de criptomoedas. Quero automatizar Tradings, fazer o sistema reconhecer padrões, comprar e vender segundo critérios que eu defina, etc. Também tenho objetivos de aprender o máximo para utilizar em automações no trabalho também, que envolve a área jurídica e trabalho em processos. Como sou fã de eletrônica e tenho cursos na área, também queria aprender o que precisa para automatizacões diversas. Existe algum curso ou trilha que me prepare com base para todas essas áreas ao mesmo tempo e a partir dele eu aprenda isoladamente aquilo que seria exigido para aplicar aos meus projetos?',
    'Bom dia, Havia pedido cancelamento de minha mensalidade no mes 2 e continuaram cobrando. Peço cancelamento da assinatura. Peço por gentileza, para efetivarem o cancelamento da assomatura e pagamento.',
    'Bom dia. Não estou conseguindo tirar os certificados dos cursos que concluí. Por exemplo, já consegui 100% no python starter, porém, não consigo tirar o certificado. Como faço?',
    'Bom dia. Não enconte no site o preço de um curso avulso. SAberiam me informar?'
]

system_message = '''Pense com cuidado ao categorizar o texto conforme as instruções.
Questões relacionadas a dúvidas de preço, sobre o produto, como funciona devem ser direciodas para "vendas".
Questões relacionadas a conta, acesso a plataforma, a cancelamento e renovação de assinatura para devem ser direciodas para "atendimento_cliente".
Questões relacionadas a dúvidas técnicas de programação, conteúdos da plataforma ou tecnologias na área da programação devem ser direciodas para "duvidas_alunos".
Mensagens suspeitas, em outras línguas que não português, contendo links devem ser direciodas para "spam".
'''

prompt = ChatPromptTemplate.from_messages([
    ('system', system_message),
    ('user', '{input}')
])

class SectorEnum(str, Enum):
    atendimento_cliente = 'atendimento_cliente'
    duvidas_aluno = 'duvidas_aluno'
    vendas = 'vendas'
    spam = 'spam'

class RedirectToSector(BaseModel):
    """Direciona a dúvida de um cliente ou aluno da escola de programação Asimov para o setor responsável"""
    sector: SectorEnum

redirect_tool = convert_to_openai_function(RedirectToSector)
chat = ChatOpenAI().bind(functions=[redirect_tool], function_call={'name': 'RedirectToSector'})

chain = (
    prompt
    | chat
    | JsonOutputFunctionsParser()
)

doubt = doubts[0]
res = chain.invoke({'input': doubt})
print('Dúvida:', doubt)
print('Resposta:', res)
