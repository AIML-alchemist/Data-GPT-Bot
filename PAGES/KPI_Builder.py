import streamlit as st
import pandas as pd
from datetime import datetime
import xlsxwriter

# Define functions for different pages
# def profile_page():
st.title("Profile Page")
name = st.text_input("Enter your name:")
job_profile = st.selectbox("Select job profile:", ["Business Analyst", "Data Scientist", "Business Executive"])
key = st.text_input("Enter the key:")
model_option = st.radio("Select a model:", ["Model 1", "Model 2"])

if st.button("Submit"):
        # Create a DataFrame to store the user details
    user_data = pd.DataFrame({
        'Name': [name],
        'Job Profile': [job_profile],
        'Key': [key],
        'Selected Model': [model_option],
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

    st.subheader("User Details:")
    st.write(f"**Name:** {name}")
    st.write(f"**Job Profile:** {job_profile}")
    st.write(f"**Key:** {key}")
    st.write(f"**Selected Model:** {model_option}")
    
    # confirm = st.checkbox("Confirm")
    updated_data = pd.DataFrame()
    if st.button("Confirm"):
        # Check if the Excel file already exists, if not, create it
        try:
            existing_data = pd.read_excel('C:/Users/Aditya/Documents/Python_Scripts/Streamlit_demo/user_details.xlsx')
            updated_data = pd.concat([existing_data, user_data], ignore_index=True)
        except FileNotFoundError:
            updated_data = user_data

        # Write the updated data to the Excel file
    updated_data.to_excel('C:/Users/Aditya/Documents/Python_Scripts/Streamlit_demo/user_details.xlsx', index=False)
    st.success("Details confirmed and saved to Excel.")

# def page2():
#     st.title("Page 2")
#     # Add content for the second page here

# # Streamlit app title


# # Sidebar navigation menu
# page = st.sidebar.radio("Go to", ["KPI_builder", "streamlit demo new"])

# # Render the selected page
# if page == "KPI_Builder":
#     profile_page()
# elif page == "Page 2":
#     page2()
