import streamlit as st
import openai

def run_request(question_to_ask):

# We use the API to submit the query
    openai.api_type = "azure"
    openai.api_base = ""
    openai.api_version = "2022-12-01"
    # openai.api_key = os.getenv("")
    # os.environ['OPENAI_API_KEY'] = ""
    engine=""
    openai.api_key = ""
    response = openai.Completion.create(
        # model_type = text-davinci-003, code-davinci-002
        engine=engine,
        prompt=question_to_ask,
        temperature=0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["plt.show()"]
        )
    res = response["choices"][0]["text"] 
    # rejig the response
    res = format_response(res)
    return res

def format_response( res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def format_question(primer_desc,primer_code , question):
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title. Set the fig subtitle as empty."
    primer_desc = primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    primer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    primer_code = primer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    primer_code = primer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    primer_code = primer_code + "df=" + "data.copy()\n"
    # st.write(primer_desc)
    # st.write("\n")
    # st.write(primer_code)
    return primer_desc,primer_code


