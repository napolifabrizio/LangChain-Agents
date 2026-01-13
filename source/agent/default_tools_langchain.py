from langchain_community.utilities.arxiv import ArxivAPIWrapper
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from agent.environs import Environs


## 1 - Criando uma tool que ja está pronta
class ArxivArgs(BaseModel):
    query:str = Field(description='Query de busca no ArXiv')

tool_arxiv = StructuredTool.from_function(
    func=ArxivAPIWrapper(top_k_results=2).run,
    args_schema=ArxivArgs,
    name='arxiv',
    description = (
        "Uma ferramenta em torno do Arxiv.org. "
        "Útil para quando você precisa responder a perguntas sobre Física, Matemática, "
        "Ciência da Computação, Biologia Quantitativa, Finanças Quantitativas, Estatística, "
        "Engenharia Elétrica e Economia utilizando artigos científicos do arxiv.org. "
        "A entrada deve ser uma consulta de pesquisa em inglês."
    ),
    return_direct=True
)

## 2 - Criando uma ferramenta que ja está pronta

from langchain_community.tools.arxiv.tool import ArxivQueryRun

tool_arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))

## 3 - Criando uma ferramenta que ja está pronta

from langchain_community.agent_toolkits.load_tools import load_tools

tools = load_tools(['arxiv'])
tool_arxiv = tools[0]

## Fazendo um teste

from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit

def routing(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        return tool_run[result.tool].run(result.tool_input)

tool_kit = FileManagementToolkit(
    root_dir='arquivos',
    selected_tools=['write_file', 'read_file', 'file_search','list_directory']
)
tools = tool_kit.get_tools()

tools_json = [convert_to_openai_function(tool) for tool in tools]
tool_run = {tool.name: tool for tool in tools}

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um assistente amigável chamado Isaac capaz de gerenciar arquivos'),
    ('user', '{input}')
])

chat = ChatOpenAI()

chain = (
    prompt
    | chat.bind(functions=tools_json)
    | OpenAIFunctionsAgentOutputParser()
    | routing
)

res = chain.invoke({'input': 'What are you able to do?'})

print(res)
