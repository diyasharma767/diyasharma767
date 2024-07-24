import streamlit as st
from main_pdf import ChatBot  # Adjust the import according to your file structure

# Initialize the chatbot
bot = ChatBot()

# Streamlit app
def main():
    st.title("Welcome")
    st.write("Ask me anything!")

    # User input
    user_query = st.text_input("Your question:")

    if user_query:
        with st.spinner('Thinking...'):
            result = bot.invoke(user_query)
        st.write("Answer:", result)

if __name__ == "__main__":
    main()
