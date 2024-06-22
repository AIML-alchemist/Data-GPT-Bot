import openai
import os
import tempfile
import streamlit as st
import pandas as pd
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, OpenAIChat , CTransformers
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import json
from sklearn import tree


#Loading the model
def load_llm(TEMP):
    # Load the locally downloaded model here
    # llm = CTransformers(
    #     model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0.5
    # )

    openai.api_type = "azure"
    openai.api_base = "https://cs-lab-azureopenai.openai.azure.com/"
    openai.api_version = "2022-12-01"
    openai.api_key = os.getenv("dfb58f8ff710406aab6350cdc9e7e38f")
    os.environ['OPENAI_API_KEY'] = "dfb58f8ff710406aab6350cdc9e7e38f"

    engine="inc-lab-AzureOpenAI-text-davinci-003"
    llm = OpenAI(openai_api_key = "dfb58f8ff710406aab6350cdc9e7e38f",engine=engine, temperature=TEMP)

    # llm = OpenAIChat(openai_api_key="api_key", temperature=0)

    return llm


def main():
    st.set_page_config(page_title="Structured Data LLM Analyser")
    st.subheader("Incedo Structured Data Analyser")
    st.write("Upload a Spreadsheet ex: CSV or XLSX")

    with st.sidebar:
        with st.expander("Settings",  expanded=True):
            TEMP = st.slider(label="LLM Temperature", min_value=0.0, max_value=1.0, value=0.5)
            st.subheader("Powered by LangChain")

    llm = load_llm(TEMP=TEMP)    

    # Upload File
    file =  st.file_uploader("Upload CSV file",type=["csv","xlsx"])
    if not file: st.stop()

    # Read Data as Pandas
    data = pd.read_csv(file)

    # Display Data Head
    st.write("Data Preview:")
    st.dataframe(data.head()) 

    def pandas_agent(input=""):
        pandas_agent_df = create_pandas_dataframe_agent(llm, data, verbose=True, openai_api_key=OPENAI_API_KEY, )
        
    def kpi_tree_builder(input=""):
        # Preprocess the dataframse        
        df=data.copy()
        le_dict = {}
        for x in zip(df.columns,df.dtypes):
            
            print(x[0],x[1])
            df[x[0]] = df[x[0]].fillna(df[x[0]].mode()[0])
            if x[1]=='object':
                le = preprocessing.LabelEncoder()
                df[x[0]] = le.fit_transform(df[x[0]])
                le_dict[x[0]]=le

        X = df.iloc[:, :-1]
        y = df.iloc[:,-1]
        dt_clf = DecisionTreeClassifier(random_state=42,max_depth=3).fit(X, y)
        #self.dt_tree_json = self.tree_to_json(self.dt_clf,list(X.columns))
        kpi_visual_tree = tree.plot_tree(dt_clf,
                   feature_names=list(df.columns)[:-1],
                   class_names=list(df.columns)[-1],
                   filled=True,impurity=False,label="root")
        # kpi_visual_tree
        plt.savefig('tree.jpg', dpi=600)
        # st.image(kpi_visual_tree[0], use_column_width=True, caption='kpi tree')
        # plt.figure()
        st.image('tree.jpg', use_column_width=True, caption='kpi tree')
    pandas_tool = Tool(
    name='Pandas Data frame tool',
    func=pandas_agent,
    description="Useful for when you need to answer questions about a Pandas Dataframe"
    )
    
    kpi_tool= Tool(
    name="KPI Tree Builder",
    func=kpi_tree_builder,
    description="UDDF|User Defined Dataframe to Load and process data from Dataframe and only Build KPI Tree"
)
    tools = [pandas_tool,kpi_tool]
    # conversational agent memory

    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=3, return_messages=True)

    # Create our agent

    chain = initialize_agent(
    #agent='chat-conversational-react-description',
    agent='zero-shot-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
    )

    agent = create_pandas_dataframe_agent(llm, data, verbose=True, openai_api_key = "dfb58f8ff710406aab6350cdc9e7e38f")
    
    
    
    def conversational_chat(query):
        # result = chain.run({"question": query, "chat_history": st.session_state['history']})
        prompt = f'''
                    Consider the uploaded pandas data, respond intelligently to user input
                    \nCHAT HISTORY: {st.session_state.history}
                    \nUSER INPUT: {query}
                    \nAI RESPONSE HERE:
                '''

                # Get answer from agent
        result = chain.run(prompt)
        # st.session_state['history'].append((query, result["answer"]))
        st.session_state.history.append(f"USER: {query}")
        st.session_state.history.append(f"AI: {result}")
        return result
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + file.name ]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="big-smile")

  
if __name__ == "__main__":

    main()   