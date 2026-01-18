from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent.environs import Environs

system_msg = """Você é um agente projetado para escrever e executar código Python para responder perguntas.
Você tem acesso a um REPL Python, que pode usar para executar código Python.
Se encontrar um erro, depure o código e tente novamente.
Use apenas a saída do seu código para responder à pergunta.
Você pode conhecer a resposta sem executar nenhum código, mas deve ainda assim executar o código para obter a resposta.
Se não parecer possível escrever código para responder à pergunta, simplesmente retorne "Não sei" como a resposta."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_msg),
    ('placeholder', '{chat_history}'),
    ('human', '{input}'),
    ('placeholder', '{agent_scratchpad}')
])

tools = [PythonAstREPLTool()]

chat = ChatOpenAI()
agent = create_tool_calling_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# res = agent_executor.invoke({"input": "Qual é o vigésimo elemento da sequência de fibonacci?"})
# print(res)


# ReAct Agent (Reason + Act) -> Usado apenas para modelos mais simples

from langchain.agents import create_react_agent
from langchain import hub

prompt = hub.pull('hwchase17/react')

print(prompt.template)
agent = create_react_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

