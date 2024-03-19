# Import the necessary libraries
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm.notebook import tqdm
import langchain
import openai
from openai import OpenAI
import string
import numpy as np
import random as rnd
import os
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

SECRET_TOKEN_openaikey = os.getenv("open_ai_key_mlbot")
SECRET_TOKEN_pineconekey = os.getenv("pinecone_key_mlbot")
SECRET_TOKEN_pineconeindex = os.getenv("pinecone_index_mlbot")
SECRET_TOKEN_GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SECRET_TOKEN_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = "5055ad72b7a3146d0"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDIIqsd8ea8t09pQIY1P2yx21gR35yhmPE"
# os.environ["GOOGLE_CSE_ID"] = SECRET_TOKEN_GOOGLE_CSE_ID
# os.environ["GOOGLE_API_KEY"] = SECRET_TOKEN_GOOGLE_API_KEY

search = GoogleSearchAPIWrapper()

tool = Tool(
    name = "Google Search",
    description = "Search Google for recent results.",
    func = search.run
)


# Python
# Agents
class Obnoxious_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the client and prompt for the Obnoxious_Agent
        self.client = openai_client

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Obnoxious_Agent
        self.prompt = f"Is the following query obnoxious: {prompt}? Answer with true or false."

    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        # ... (code for extracting action)
        message_content = response.choices[0].message.content
        if "true" in message_content or "True" in message_content:
            return True
        else:
            return False
    
    def check_query(self, query):
        # TODO: Check if the query is obnoxious or not
        # ... (code for checking query)
        self.set_prompt(query)
        message = {"role": "user", "content": self.prompt}
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message]
        )
        return self.extract_action(response)
    
class Query_Agent:
    def __init__(self, pinecone_vector_store, openai_client, embeddings) -> None:
        # TODO: Initialize the Query_Agent agent
        self.vector_store = pinecone_vector_store
        self.client = openai_client
        self.embeddings = embeddings

    def query_vector_store(self, conv_history,query, top_k=5):
        # TODO: Query the Pinecone vector store
        message = ' '.join([item['content'] for item in conv_history]) + ' ' + query
        rel_doc_agent = Relevant_Documents_Agent(self.vector_store,self.client, self.embeddings)
        context = rel_doc_agent.get_relevance(message)
        #query_embedding = self.embeddings.embed_query(query)
        return context

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Query_Agent agent
        self.prompt = f"Is the following query a general greeting with no question? Answer with true or false. Query: {prompt}"
        
    def extract_action(self,query = None):
        # TODO: Extract the action from the response
        self.set_prompt(query)
        message = [{"role": "user", "content": self.prompt}]
        response = self.client.chat.completions.create(
            model=st.session_state['openai_model'],
            messages=message
                )
        message_content = response.choices[0].message.content
        print(f"Message content: {message_content}")
        if "true" in message_content or "True" in message_content or "yes" in message_content or "Yes" in message_content:
            return True
        else:
            return False

class Answering_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the Answering_Agent
        self.client = openai_client
    #  def generate_response(self, query, docs, conv_history, k=5):
    def generate_response(self, query,conv_history):
        # TODO: Generate a response to the user's query
        #model="gpt-3.5-turbo"
        message = conv_history + [{"role": "user", "content": query}]
        response = self.client.chat.completions.create(
            model=st.session_state['openai_model'],
            messages=message
                )
        return response

class Relevant_Documents_Agent:
    def __init__(self, vector_store, openai_client, embeddings) -> None:
        # TODO: Initialize the Relevant_Documents_Agent
        self.client = openai_client
        self.embeddings = embeddings
        self.vector_store = vector_store

    def get_relevance(self, conversation) -> str:
        # TODO: Get if the returned documents are relevant
        query_embedding = self.embeddings.embed_query(conversation)
        results = self.vector_store.query(vector=[query_embedding],top_k=5,include_metadata=True, namespace='chunks-500')
        scores = [match['score'] for match in results['matches']]
        avg_score = np.mean(scores)
        print(f"Average score: {avg_score}")
        if avg_score > 0.78:
            texts = [match['metadata']['text'] for match in results['matches']]
            context = " ".join(texts)#
            return "Relevant Context: " + context
        else:
            return "No relevant documents found"

class Topic_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the client and prompt for the Obnoxious_Agent
        self.client = openai_client

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Obnoxious_Agent
        self.prompt = f"Is the following query about machine learning and artificial intelligence: {prompt}? Answer with true or false."

    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        # ... (code for extracting action)
        message_content = response.choices[0].message.content
        if "true" in message_content or "True" in message_content:
            return True
        else:
            return False
    
    def check_query(self, query, conv_history, binary=False):
        # ... (code for checking query)
        if binary == False:
            self.set_prompt(query)
        else:
            self.prompt = f"Is the query related to the conversation? Answer: True or False. Query: {query}. Conversation: {conv_history[-4:]}"
        message = {"role": "user", "content": self.prompt}
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message]
        )
        return self.extract_action(response)


class GoogleSearch_Agent:
     def __init__(self, tool) -> None:
         self.tool = tool
        
     def search(self, query):
         return self.tool.run(query)
     
         
        
    

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name,tool) -> None:
        # TODO: Initialize the Head_Agent
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name
        self.tool = tool
        self.openai_client = OpenAI(api_key=openai_key)
        self.pc = Pinecone(api_key=pinecone_key)
        self.vector_store = self.pc.Index(name=pinecone_index_name)
        self.embeddings = OpenAIEmbeddings(api_key=openai_key)
        # print(f"Index info: \n: {self.vector_store.describe_index_stats()}")
        
    def setup_sub_agents(self):
        # TODO: Setup the sub-agents
        self.obnoxious_agent = Obnoxious_Agent(self.openai_client)
        self.relevant_documents_agent = Relevant_Documents_Agent(self.vector_store,self.openai_client,self.embeddings)
        self.answering_agent = Answering_Agent(self.openai_client)
        self.googlesearch_agent = GoogleSearch_Agent(self.tool)
        self.topic_agent = Topic_Agent(self.openai_client)

        self.query_agent = Query_Agent(self.vector_store, self.openai_client, OpenAIEmbeddings(api_key=self.openai_key))
        



    def main_loop(self, prompt,conv_history,toggle_google_search_agent=False):
        # TODO: Run the main loop for the chatbot
        self.setup_sub_agents()
        if self.obnoxious_agent.check_query(prompt) == True:
            prompt = "Answer that the query is obnoxious and cannot be answered.Stay polite."
            response = self.answering_agent.generate_response(prompt,conv_history)
            response = response.choices[0].message.content
            print("Obnoxious query detected")
        else:
            if self.query_agent.extract_action(prompt) == True:
                prompt = "Answer to the user in a polite manner and tell them that you are specialized on topics related to machine learning and artificial intelligence."
                response = self.answering_agent.generate_response(prompt,conv_history)
                response = response.choices[0].message.content
                print("General greeting detected")
            else:
                off_topic_prompt = self.topic_agent.check_query(prompt,conv_history[-4:],False)
                off_topic_conv = self.topic_agent.check_query(prompt,conv_history[-4:],True)
                if off_topic_prompt == True:
                    context = self.query_agent.query_vector_store(conv_history[-4:],prompt)
                    if context != "No relevant documents found":
                        prompt = f"Given this {context}. Answer the following query and take into account the given context. Query: {prompt}"
                        response = self.answering_agent.generate_response(prompt,conv_history)
                        response = response.choices[0].message.content
                        print("Relevant documents found")
                    # else:
                    #     prompt = "Answer in a polite manner that no relevant documents were found and that the user should ask questions about the book with the title Machine Learning."
                    #     response = self.answering_agent.generate_response(prompt,conv_history)
                    #     response = response.choices[0].message.content
                    else:
                        if toggle_google_search_agent == True:
                            googlecontext = self.googlesearch_agent.search(prompt)
                            print(f"Google context: {googlecontext[:500]}")
                            prompt = f" Given the following context, answer the query in the context of machine learning and artificial intelligence. Start with saying that you had to do a google search.  Context: {googlecontext}. Query: {prompt}."
                            response = self.answering_agent.generate_response(prompt,conv_history)
                            response = response.choices[0].message.content
                            print("No relevant documents found but google search was done")
                        else:
                            prompt = "Answer in a polite manner that no relevant information were found in the book. Include this sentence: You can enable the google search agent to enhance my knowledge and ask again."
                            response = self.answering_agent.generate_response(prompt,conv_history)
                            response = response.choices[0].message.content
                            print("No relevant documents found")

                elif off_topic_prompt == False and off_topic_conv == True:
                    context = self.query_agent.query_vector_store(conv_history[-4:],prompt)
                    if context != "No relevant documents found":
                        prompt = f"Given this {context}. Answer the following query and take into account the given context. Query: {prompt}"
                        response = self.answering_agent.generate_response(prompt,conv_history)
                        response = response.choices[0].message.content
                        print("Relevant documents found-2")
                    # else:
                    #     prompt = "Answer in a polite manner that no relevant documents were found and that the user should ask questions about the book with the title Machine Learning."
                    #     response = self.answering_agent.generate_response(prompt,conv_history)
                    #     response = response.choices[0].message.content
                    else:
                        if toggle_google_search_agent == True:
                            googlecontext = self.googlesearch_agent.search(prompt)
                            print(f"Google context: {googlecontext[:500]}")
                            prompt = f" Given the following context, answer the query in the context of machine learning and artificial intelligence. Start with saying that you had to do a google search, because you did not find the enough relevant information in the book.  Context: {googlecontext}. Query: {prompt}."
                            response = self.answering_agent.generate_response(prompt,conv_history)
                            response = response.choices[0].message.content
                            print("No relevant documents found but google search was done-2")
                        else:
                            prompt = "Answer in a polite manner that no relevant information were found in the book and that the user could enhance the search with google by enabling the agent and asking again."
                            response = self.answering_agent.generate_response(prompt,conv_history)
                            response = response.choices[0].message.content
                            print("No relevant documents found-2")
                elif off_topic_prompt == False and off_topic_conv == False:
                        prompt = "Answer in a polite manner that you cannot answer the question and tell the user that you are specialized on topics related to machine learning and artificial intelligence."
                        response = self.answering_agent.generate_response(prompt,conv_history)
                        response = response.choices[0].message.content
                        print("Off-topic query detected")

        return response
    


st.title("The Pytorch Professor: Your ML Book Chat Bot")


# TODO: Replace with your actual OpenAI API key
# Open the file
openai_key = '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'
pinecone_key = "85c03cc5-ce19-4478-84a1-66db09828e53"
pinecone_index_name = "eep596llm"
# TODO: Replace with your actual OpenAI API key
# Open the file
# openai_key = SECRET_TOKEN_openaikey
# pinecone_key = SECRET_TOKEN_pineconekey
# pinecone_index_name = SECRET_TOKEN_pineconeindex


# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation():
    # ... (code for getting conversation history)
    return [{'role': 'system', 'content': 'You are talking to GPT-3.5-turbo.'}] + st.session_state['messages']


Head_Agent = Head_Agent(openai_key, pinecone_key, pinecone_index_name,tool)
#Head_Agent.setup_sub_agents()



# Check for existing session state variables
if "openai_model" not in st.session_state:
    # ... (initialize model)
    st.session_state['openai_model'] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    # ... (initialize messages)
    st.session_state['messages'] = []

# Check for existing session state variables
if "google_search_agent_enabled" not in st.session_state:
    # Initialize Google search agent state
    st.session_state['google_search_agent_enabled'] = False

# Add a button to toggle Google search agent
if st.button('Toggle Google Search Agent'):
    # Toggle the state of the Google search agent
    st.session_state['google_search_agent_enabled'] = not st.session_state['google_search_agent_enabled']

# Display the state of the Google search agent
st.write(f"Google Search Agent is {'enabled' if st.session_state['google_search_agent_enabled'] else 'disabled'}")


# Display existing chat messages
# ... (code for displaying messages)
for message in st.session_state['messages']:
    # if message['role'] == 'user':
    #     st.text_area("You", value=message['content'], key=message['content']+"-"+f"{len(st.session_state['messages'])}")
    # else:
    #     st.text_area("AI", value=message['content'], key=message['content']+"-AI-"+f"{len(st.session_state['messages'])}")
    if message['role'] == 'user':
        st.text_area("You", value=message['content'], key=message['content']+f"{rnd.randint(0,1000)}")
    else:
        st.text_area("AI", value=message['content'], key=message['content']+f"{rnd.randint(0,1000)}")

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # ... (append user message to messages)
    st.session_state['messages'].append({'role': 'user', 'content': prompt})

    # ... (display user message)
    st.text_area("You", value=prompt, key=len(st.session_state['messages']))

    # Generate AI response
    with st.chat_message("assistant"):
        # ... (send request to OpenAI API)
        if st.session_state['google_search_agent_enabled'] == True:
            response = Head_Agent.main_loop(prompt,get_conversation(),True)
        else:
            response = Head_Agent.main_loop(prompt,get_conversation(),False)
        # response = client.chat.completions.create(
        #     model=st.session_state['openai_model'],
        #     messages=get_conversation(),
        # )
        # ... (get AI response and display it)
        st.text_area("AI", value=response, key=f"AI-{len(st.session_state['messages'])}")


    # ... (append AI response to messages)
    st.session_state['messages'].append({'role': 'system', 'content': response})