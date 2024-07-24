# from main import ChatBot
# import streamlit as st

# bot = ChatBot()

# st.set_page_config(page_title="Random Fortune Telling Bot")
# with st.sidebar:
#     st.title('Random Fortune Telling Bot')

# # Function for generating LLM response
# def generate_response(input):
#     result = bot.rag_chain.invoke(input)
#     return result

# # Store LLM generated responses
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's unveil your future"}]

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # User-provided prompt
# if input := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": input})
#     with st.chat_message("user"):
#         st.write(input)

# # Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Getting your answer from mystery stuff.."):
#             response = generate_response(input)
#             st.write(response)
#     message = {"role": "assistant", "content": response}
#     st.session_state.messages.append(message)
import streamlit as st
from main import ChatBot

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

