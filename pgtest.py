import os
import psycopg2
from dotenv import dotenv_values
from openai import AzureOpenAI
import time
import streamlit as st
import  time
import config
import json
import os
import sys
import uuid
import datetime
import glob
import time
import uuid
import re
import csv
import string
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import dotenv_values
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from dotenv import dotenv_values
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
from langchain_openai.embeddings import OpenAIEmbeddings

env_name = "example.env" # following example.env template change to your own .env file name
config = dotenv_values(env_name)

# Connect to PostgreSQL

openai_endpoint = config['openai_endpoint']
openai_key = config['openai_key']
openai_version = config['openai_version']
openai_chat_model = config['AZURE_OPENAI_CHAT_MODEL']
openai_embeddings_model = config['openai_embeddings_deployment']

openai_client = AzureOpenAI(
  api_key = openai_key,  
  api_version = openai_version,  
  azure_endpoint =openai_endpoint 
)


def clearcache():
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM tablecahedoc')
    conn.commit()
    cur.close()
    conn.close()
    
def intialize():
    
# Connect to PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()

    # Execute a command: this creates a new table

    cur.execute('CREATE TABLE tablecahedoc (id serial PRIMARY KEY,'
                                 'prompt text  NOT NULL,'
                                 'completion text NOT NULL,'
                                 'completiontokens integer NOT NULL,'
                                 'promptTokens integer NOT NULL,'
                                 'totalTokens integer NOT NULL,'
                                 'model varchar(150) NOT NULL,'   
                                 'usname text NOT NULL,'
                                 'date_added date DEFAULT CURRENT_TIMESTAMP);'
        )


    cmd = """ALTER TABLE tablecahedoc  ADD COLUMN dvector vector(3072)  GENERATED ALWAYS AS ( azure_openai.create_embeddings('text-embedding-3-large', prompt)::vector) STORED; """
    cur.execute(cmd)

   # cm = """CREATE INDEX tablecahedoc_embedding_diskann_idx ON tablecahedoc USING diskann (dvector vector_cosine_ops)"""
   # cur.execute(cm)

    cur.execute('CREATE TABLE data (id serial PRIMARY KEY,'
                                 'filename text NOT NULL,'
                                 'typefile text NOT NULL,' 
                                 'chuncks text,'
                                 'date_added date DEFAULT CURRENT_TIMESTAMP);'
                                 )

    
    cmd = """ALTER TABLE data  ADD COLUMN dvector vector(3072)  GENERATED ALWAYS AS ( azure_openai.create_embeddings('text-embedding-3-large', chuncks)::vector) STORED; """
    cur.execute(cmd)

   # cmd2 = """CREATE INDEX data_embedding_diskann_idx ON data USING diskann (dvector vector_cosine_ops)"""
    #cur.execute(cmd2)
    # Commit the transaction
    conn.commit()

# Close the cursor and connection
    cur.close()
    conn.close()

def cleanall():

# Connect to PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()

    # Execute a command: this creates a new table

    cur.execute('DROP TABLE IF EXISTS tablecahedoc;')
    conn.commit()
    cur.execute('DROP TABLE IF EXISTS data;')
    conn.commit()
    cur.close()
    conn.close()

def loadpdffile(name,file) :
    
   
    loader = PyPDFLoader(file)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    
    try:
        for d in docs : 
            data = str(d)
            print(data)
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                        'VALUES (%s, %s,%s)',
                        (name,"pdf",data)
                        )
            conn.commit()
            cur.close()
            conn.close()
    except : 
     raise     

def loadwordfile(name,file) :
    
   

    
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file)
    data = loader.load()    
  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
   
  
    
    try:
      for d in docs : 
            data = str(d)
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                        'VALUES (%s, %s,%s)',
                        (name,"word",data)
                        )
            conn.commit()
            cur.close()
            conn.close()
    except : 
     raise    
 
 
def loadjsonfile(name,file): 
    
    with open(file,encoding="utf8") as file:
        docu = json.load(file)
        for row in docu:
            data = json.dumps(row)
            print (data)
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                'VALUES (%s, %s,%s)',
                (name,"json",data)
               )
            conn.commit()
            cur.close()
            conn.close()
 
 
def loadcsvfile(name,file) :
    
 with open(file, mode='r', encoding='utf-8-sig') as file:
  csv_reader = csv.DictReader(file)
  for row in csv_reader:
    data = json.dumps(row)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                'VALUES (%s, %s,%s)',
                (name,"csv",data)
               )
    conn.commit()
    cur.close()
    conn.close()
      
def generate_embeddings(openai_client, text):
    """
    Generates embeddings for a given text using the OpenAI API v1.x
    """
    
    response = openai_client.embeddings.create(
        input = text,
        model= openai_embeddings_model
    
    )
    embeddings = response.data[0].embedding
    return embeddings
 
 
def get_completion(openai_client, model, prompt: str):    
   
    response = openai_client.chat.completions.create(
        model = model,
        messages =   prompt,
        temperature = 0.25
        
    )   
    
    return response.model_dump()

def cacheresponse(user_prompt,  response , name):

    
        
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO tablecahedoc (prompt, completion, completiontokens, promptTokens,totalTokens,usname, model)'
                    'VALUES (%s, %s, %s, %s ,%s, %s,%)',
                    (user_prompt, response['choices'][0]['message']['content'], response['usage']['completion_tokens'], response['usage']['prompt_tokens'],response['usage']['total_tokens'],name, response['model']))
    


    print("item inserted into cache.")
    conn.commit()
    cur.close()
    conn.close()

def get_db_connection():
    conn = psycopg2.connect(
        dbname=config['pgdbname'],
        user=config['pguser'],
        password=config['pgpassword'],
        host=config['pghost'],
        port=config['pgport'])
    return conn

def authenticate(username):
    # Pour des raisons de démonstration, nous utilisons une vérification simple
    return username 
    
def generatecompletionede(user_prompt ) -> str:
    
 
    system_prompt = '''
    You are an intelligent assistant for yourdata , please answer in the same langage use by the user . You are designed to provide helpful answers to user questions about your data.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.
        - Only answer questions related to the information provided below. 
        - Write two lines of whitespace between each answer in the list.'''
        
    # system prompt

    messages = [{'role': 'system', 'content': system_prompt}]
    #user prompt
    messages.append({'role': 'user', 'content': user_prompt})
    
    vector_search_results =  ask_dbvector(user_prompt )
    
    for result in vector_search_results:
     
        messages.append({'role': 'system', 'content': result})
    
    response = get_completion(openai_client, openai_chat_model, messages)

    return response

def cacheresponse(user_prompt,  response , name):

    
        
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO tablecahedoc (prompt, completion, completiontokens, promptTokens,totalTokens, model,usname)'
                    'VALUES (%s, %s, %s, %s ,%s, %s,%s)',
                    (user_prompt, response['choices'][0]['message']['content'], response['usage']['completion_tokens'], response['usage']['prompt_tokens'],response['usage']['total_tokens'], response['model'] ,name))
    


    print("item inserted into cache.")
    conn.commit()
    cur.close()
    conn.close()

def cachesearch(test,name):
    conn = get_db_connection()
    cur = conn.cursor()
   
    query = f"""SELECT e.completion
    FROM tablecahedoc e  where e.usname = '""" + str(name) +"""' and e.dvector <=> azure_openai.create_embeddings('text-embedding-3-large', ' """ + str(test) + """')::vector < 0.07  ORDER BY  e.dvector <=> azure_openai.create_embeddings('text-embedding-3-large','""" + str(test) +"""')::vector  LIMIT 1;"""
   
    cur.execute(query)
    resutls = cur.fetchall()
   

  
    return resutls
    
def  ask_dbvector(textuser):
    
    conn = get_db_connection()
    cur = conn.cursor()
  
    
    query = f"""SELECT
     e.chuncks , e.filename
    FROM data e where e.dvector <=> azure_openai.create_embeddings('text-embedding-3-large', ' """ + str(textuser) + """')::vector < 0.25  ORDER BY  e.dvector <=> azure_openai.create_embeddings('text-embedding-3-large','""" + str(textuser) +"""')::vector  LIMIT 4;"""
    
    print(query)
    cur.execute(query)
    resutls = str(cur.fetchall())

                       
    chars = re.escape(string.punctuation)
    res = re.sub('['+chars+']', '',resutls)                        
                         
  
    return res



def chat_completion(user_input,username ):


    # Query the chat history cache first to see if this question has been asked before
    cache_results = cachesearch(user_input ,username)

    if len(cache_results) > 0:
       
        return cache_results[0], True
    else:
        # Perform vector search on the movie collection
       
   
        
        # Generate the completion
        completions_results = generatecompletionede(user_input)

        # Cache the response
        cacheresponse(user_input, completions_results,username)

        
        
        return completions_results['choices'][0]['message']['content'], False


# Application Streamlit
def main():
    st.title("Connection page with Postgreqsl sample ")
   
    global chat_history
    chat_history = []
  
    # Initialize session state for login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
       
        username = st.session_state.username
        display = "Welcome to the applicaton: " + username 
        st.success(display)

        # Onglets
         # Onglets
        tab1, tab2, tab3, tab4  = st.tabs(["Configuration", "Loading file", "Chat with your data","made by " ])


        with tab1:
            st.header("Configuration")
            
            if st.button("create the table for data and cache "):
                st.write("start the operation")
                intialize()
                st.write("the database and collection vector and cache are created ")
                
            
            if st.button("clear the cache table "):
                st.write("start clear the Cache")
                clearcache()
                st.write("Cache cleared.")
                
           
            if st.button("delete all table to reinit"):
                st.write("delete all collection")
                cleanall()    
                st.write("all collection delete ")
            
          
  

        with tab2:
            st.header("Chargement de document ")
        
            uploaded_file = st.file_uploader("Choisissez un fichier", type=["pdf", "docx","csv",  "json"])
            if uploaded_file is not None:
                st.write("Fichier sélectionné:", uploaded_file.name)
        
            # Enregistrer temporairement le fichier téléchargé pour obtenir le chemin absolu
                with open(uploaded_file.name, "wb") as f:
                     f.write(uploaded_file.getbuffer())

            # Obtenir le chemin absolu du fichier
                absolute_file_path = os.path.abspath(uploaded_file.name)
                st.write(f"Le chemin absolu du fichier est : {absolute_file_path}")
                
                
                if st.button("load data "):
                    st.write("start the operation")
                    
                
                    if ".doc" in uploaded_file.name:
                        st.write("Le fichier est un document pdf"+ uploaded_file.name )
                        name = uploaded_file.name.replace('.doc', '')
                        loadwordfile(name,absolute_file_path )
                        st.write("Le fichier est charge" +uploaded_file.name )
                    
                        st.write("Le fichier est un document Word.")
                    elif ".pdf" in uploaded_file.name:
                        st.write("Le fichier est un document pdf"+ uploaded_file.name )
                        name = uploaded_file.name.replace('.pdf', '')
                        loadpdffile(name,absolute_file_path )
                        st.write("Le fichier est charge" +uploaded_file.name )
                    
                    elif ".json" in uploaded_file.name:
                        st.write("Le fichier est un document csv"+ uploaded_file.name )
                        name = uploaded_file.name.replace('.json', '')
                        loadjsonfile(name,uploaded_file.name)
                        st.write("Le fichier est charge" +uploaded_file.name )
                    
                        
                        st.write("Le fichier est un document JSON." + uploaded_file.name )
                    elif ".csv" in uploaded_file.name:
                        st.write("Le fichier est un document csv"+ uploaded_file.name )
                        name = uploaded_file.name.replace('.csv', '')
                        loadcsvfile(name,uploaded_file.name)
                        st.write("Le fichier est charge" +uploaded_file.name )
                    
                    os.remove(absolute_file_path)
                    st.write(f"Le fichier temporaire {absolute_file_path} a été supprimé.")

         
        with tab3:
            st.header("Chat")
            reset_button_key = "reset_button"
            reset_button = st.button("Reset Chat",key=reset_button_key)
            if reset_button:
                st.session_state.conversation = None
                st.session_state.chat_history = None
                
            st.write("Chatbot goes here")
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
                ]
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
           
                          
            if prompt := st.chat_input(placeholder="groupama"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    question = prompt.replace("""'""", '')
                    start_time = time.time()
                    response_payload, cached = chat_completion(question,username)
                    end_time = time.time()
                    elapsed_time = round((end_time - start_time) * 1000, 2)
                    response = response_payload
              
                    details = f"\n (Time: {elapsed_time}ms)"
                    if cached:
                        details += " (Cached)"
                        chat_history.append([question, str(response[0]) + "for "+ username + details])
                    else:
                        chat_history.append([question, response + details])
        
                    st.session_state.messages.append({"role": "assistant", "content":chat_history})
                    st.write(chat_history)
            
   
            
        with tab4:
            st.write("made by emmanuel deletang in case of need contact him at edeletang@microsoft.com")
                
        
    else:
        # Formulaire de connexion
      
        username_input = st.text_input("Nom d'utilisateur")
       

        if st.button("Connexion"):
            if authenticate(username_input):
                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")


if __name__ == "__main__":
    print("main")
    main()