from langchain_community.utilities.arxiv import ArxivAPIWrapper
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough

from agent.environs import Environs

from agent.temperature_tool import (
    tools, tool_run, json_tools, chat
)

pass_through = RunnablePassthrough.assign(
    agent_scratchpad = lambda x: format_to_openai_function_messages(x['intermediate_steps'])
)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um assistente amigável chamado Isaac'),
    ('user', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

agent_chain = pass_through | prompt | chat.bind(functions=json_tools) | OpenAIFunctionsAgentOutputParser()

def run_agent(input):
    steps = []
    while True:
        response = agent_chain.invoke({
            'input': input,
            'intermediate_steps': steps
        })
        if isinstance(response, AgentFinish):
            return response
        obs = tool_run[response.tool].run(response.tool_input)
        steps.append((response, obs))

# print(run_agent("Qual a temperatura no Rio de Janeiro?"))

## jeito resumido, sem precisar criar o metodo run_agent e adicionando memoria ao modelo

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

pass_through = RunnablePassthrough.assign(
    agent_scratchpad = lambda x: format_to_openai_function_messages(x['intermediate_steps'])
)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um assistente amigável chamado Isaac'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad'),
])

agent_chain = pass_through | prompt | chat.bind(functions=json_tools) | OpenAIFunctionsAgentOutputParser()

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key='chat_history'
)

agent_executor = AgentExecutor(
    agent=agent_chain,
    tools=tools,
    memory=memory,
    # verbose=True
)

# print(agent_executor.invoke({"input": "Olá, meu nome é fabrizio"})["output"])
# print(agent_executor.invoke({"input": "Olá, qual é meu nome?"})["output"])
# print(agent_executor.invoke({"input": "Olá, qual a temperatura em São Paulo?"})["output"])

# print(prompt.model_json_schema())