import openai
import os
import tempfile
import streamlit as st
import pandas as pd
import seaborn as sns
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
import numpy as np
from vis_plotter import run_request, format_question, format_response, get_primer
from langchain import LLMChain,PromptTemplate
import matplotlib.pyplot as plt

def convert_to_dict_or_keep_original(input_string):
    if ':' in str(input_string):
        key, value = input_string.split(':', 1)
        value = value.strip()
        return value
    else:
        input_string = input_string.strip()
        return input_string

def check_and_print_saved_plot():
    # Check if any plot files exist
    plot_files = [file for file in os.listdir() if file.startswith('plot_') and file.endswith('.png')]
    
    if plot_files:
        # Print and delete each plot file
        for plot_file in plot_files:
            # print(f"Found saved plot: {plot_file}")
            plt.figure()
            st.image(plot_files, use_column_width=True, caption='visualisation plot')
            plt.imshow(plt.imread(plot_file))
            plt.show()
            os.remove(plot_file)


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
    openai.api_base = ""
    openai.api_version = "2022-12-01"
    openai.api_key = os.getenv("")
    os.environ['OPENAI_API_KEY'] = ""

    engine=""
    llm = OpenAI(openai_api_key = "",engine=engine, temperature=TEMP)

    return llm


def main():
    st.set_page_config(page_title="Structured Data LLM Analyser")
    st.subheader("OpenAI LangChain Structured Data Bot")
    st.write("Upload a CSV or XLSX file and query answers from your data.")

    with st.sidebar:
        with st.expander("Settings",  expanded=True):
            TEMP = st.slider(label="LLM Temperature", min_value=0.0, max_value=1.0, value=0.5)
        file =  st.file_uploader("Upload CSV file",type=["csv","xlsx"])
    llm = load_llm(TEMP=TEMP)    

    # Upload File
    if not file: st.stop()

    # Read Data as Pandas
    data = pd.read_csv(file)
    # pandas_ai = PandasAI(llmpai, conversational=False)
    # sdf = SmartDataframe(data, config={"llm": llmpai})

    # Display Data Head
    st.write("Data Preview:")
    st.dataframe(data.head()) 

    def pandas_agent(input=""):
        pandas_agent_df = create_pandas_dataframe_agent(llm, data, verbose=True, openai_api_key=OPENAI_API_KEY, )
    
    # conversational agent memory
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=3, return_messages=True)

    #return_intermediate_steps=True,
    agent = create_pandas_dataframe_agent(llm, data, verbose=True,  openai_api_key = "dfb58f8ff710406aab6350cdc9e7e38f")

    def conversational_chat(query):

        input_prompt = '''The user input is a question over a dataset fed by the user to an Data Scientist AI bot and in return the user can ask it 
                        either to generate outputs in form of string/text or a visualisation plot or both.
                        Seeing this user input question: {query} you have to reply either Text, Plot or Both. Your answer must not contain any other thing and must
                        reply as either Text, Plot or Both  
                        
                    '''
        
        prompt = PromptTemplate(template=input_prompt, input_variables=["query"])

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response_type = llm_chain.run(query)
        print(str(response_type))
        # key, value = str(response_type).split(':')
        # Create a dictionary
        # my_dict = {key.strip(): value.strip()}
        # value = value.strip()
        # print(value)
        value = convert_to_dict_or_keep_original(response_type)
        print(value)
        result=""
        if value == 'Text':
            
            prompt = f'''You are an intelligent data scientist. You have to study the data you are provided with and then 
                    help the user with its query.
                    \nCHAT HISTORY: {st.session_state.history}
                    \nUSER INPUT: {query}
                    you have to generate only a text based answer to that query and make sure that being a data sceintist 
                    your answers must be very detailed, have very well calculated numbers as well as explanations too. Answers can be divided into bullet points 
                    or paragraphs too if required. Also if possible adding a hypothesis/scenario based example based upon the data will be much appreciated by the user.
                    \nAI RESPONSE HERE:
                    '''
            result = agent.run(prompt)
            # st.session_state.history.append(f"USER: {query}")
            # st.session_state.history.append(f"AI: {result}")

        elif value == 'Plot':
            print('using plot generator')
            try:

                primer1,primer2 = get_primer(data)
                # Format the question
                question_to_ask = format_question(primer1,primer2 , query)  
                result = run_request(question_to_ask)
                            # the answer is the completed Python script so add to the beginning of the script to it.
                result = primer2 + result
                # st.write(answer)
                plot_area = st.empty()
                print(result)
                plot_area.pyplot(exec(result))    
                # st.session_state.history.append(f"USER: {query}")
                # st.session_state.history.append(f"AI: {result}")        
            except:
                prompt = f'''
                    You are an intelligent data scientist who is working with tabular data sets.
                    Consider the uploaded pandas data, respond intelligently to user input
                    \nCHAT HISTORY: {st.session_state.history}
                    \nUSER INPUT: {query}
                    you have to first generate the plot as per user requirement then make sure that use plt.savefig() command to save the graph as image 
                    with image name as plot_graph and in png format.Make sure you don't forget to save the plot once its generated. Now once the plot has been saved re refer 
                    to the user input question and see that kind of analysis or hypothesis or insight or text based answer the user wants and depending on that generate 
                    the suitable answer which must be self explanatory and elaborate.
                    Also for every plot generate some human 
                    understandable insights as output.
                    \nAI RESPONSE HERE:
                    '''
                
                result = agent.run(prompt)


        elif value == 'Both':
            try:

                prompt = f'''
                        You are an intelligent data scientist who is working with tabular data sets.
                        Consider the uploaded pandas data, respond intelligently to user input
                        \nCHAT HISTORY: {st.session_state.history}
                        \nUSER INPUT: {query}
                        You have to first generate the plot as per user requirement then make sure that use plt.savefig() command to save the graph as image 
                        with image name as plot_graph and in png format.Make sure you don't forget to save the plot once its generated. Now once the plot has been saved re refer 
                        to the user input question and see that kind of analysis or hypothesis or insight or text based answer the user wants and depending on that generate 
                        the suitable answer which must be self explanatory and elaborate.
                        Also for every plot generate some human 
                        understandable insights as output and keep them detailed.
                        \nAI RESPONSE HERE:
                    '''
                result = agent.run(prompt)

            except:

                result = "Please elborate and clarify your question again."


        else:
            print('using general tool')
            new_prompt = f'''
                    You are an intelligent data scientist who is working with tabular data sets.
                    Consider the uploaded pandas data, respond intelligently to user input
                    \nCHAT HISTORY: {st.session_state.history}
                    \nUSER INPUT: {query}
                    

                    Now you have 3 instructions:
                    1 If {response_type} = Text, then you have to generate only a text based answer to that query and make sure that being a data sceintist 
                    your answers must be very detailed, have very well calculated numbers as well as explanations too. Answers can be divided into bullet points 
                    or paragraphs too if required. Also if possible adding a hypothesis/scenario based example based upon the data will be much appreciated by the user.
                    2. If {response_type} = Plot, then you have to plot it as per user requirement then make sure that use plt.savefig() command to save the graph as image 
                    with image name as plot_graph and in png format.Make sure you don't forget to save the plot once its generated.
                    3. If {response_type} = Both, then you have to first generate the plot as per user requirement then make sure that use plt.savefig() command to save the graph as image 
                    with image name as plot_graph and in png format.Make sure you don't forget to save the plot once its generated. Now once the plot has been saved re refer 
                    to the user input question and see that kind of analysis or hypothesis or insight or text based answer the user wants and depending on that generate 
                    the suitable answer which must be self explanatory and elaborate.
                    Also for every plot generate some human 
                    understandable insights as output.
                    \nAI RESPONSE HERE:
                '''
            result = agent.run(new_prompt)
        # import json

        # print(json.dumps(result["intermediate_steps"], indent=2))
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
            check_and_print_saved_plot()
            st.session_state['generated'].append(output)
            
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                # plot_image = cached_plot()
                # Display the cached image using st.image
                
                # st.image(plot_image, use_column_width=True, caption='loan_amount_credit_history.png')

  
if __name__ == "__main__":

    main()   
    
