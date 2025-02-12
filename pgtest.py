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
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader

env_name = "example.env" # following example.env template change to your own .env file name
config = dotenv_values(env_name)



def clearcache(dbname,user,password,host,port):
    
    conn = get_db_connection(dbname,user,password,host,port)
    cur = conn.cursor()
    cur.execute('DELETE FROM tablecahedoc')
    conn.commit()
    cur.close()
    conn.close()
    
def intialize(dbname,user,password,host,port,embeddingssize,openai_embeddings_model):
    
# Connect to PostgreSQL
    conn = get_db_connection(dbname,user,password,host,port)
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


    cmd = """ALTER TABLE tablecahedoc  ADD COLUMN dvector vector("""+str(embeddingssize)+""")  GENERATED ALWAYS AS ( azure_openai.create_embeddings('"""+ str(openai_embeddings_model)+"""', prompt)::vector) STORED; """
    cur.execute(cmd)

   # cm = """CREATE INDEX tablecahedoc_embedding_diskann_idx ON tablecahedoc USING diskann (dvector vector_cosine_ops)"""
   # cur.execute(cm)

    cur.execute('CREATE TABLE data (id serial PRIMARY KEY,'
                                 'filename text NOT NULL,'
                                 'typefile text NOT NULL,' 
                                 'chuncks text,'
                                 'date_added date DEFAULT CURRENT_TIMESTAMP);'
                                 )

    
    cmd = """ALTER TABLE data  ADD COLUMN dvector vector("""+str(embeddingssize)+""")  GENERATED ALWAYS AS ( azure_openai.create_embeddings('"""+ str(openai_embeddings_model)+"""', chuncks)::vector) STORED; """
    cur.execute(cmd)

   # cmd2 = """CREATE INDEX data_embedding_diskann_idx ON data USING diskann (dvector vector_cosine_ops)"""
    #cur.execute(cmd2)
    # Commit the transaction
    conn.commit()

# Close the cursor and connection
    cur.close()
    conn.close()

def cleanall(dbname,user,password,host,port):

# Connect to PostgreSQL
    conn = get_db_connection(dbname,user,password,host,port)
    cur = conn.cursor()

    # Execute a command: this creates a new table

    cur.execute('DROP TABLE IF EXISTS tablecahedoc;')
    conn.commit()
    cur.execute('DROP TABLE IF EXISTS data;')
    conn.commit()
    cur.close()
    conn.close()



def loadpptfile(name,file,dbname,user,password,host,port) :
    
    loader = UnstructuredPowerPointLoader(file)
    data = loader.load()
    
    print(data)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
   
   
    try:
        for d in docs : 
            data = str(d)
            print(data)
            conn = get_db_connection(dbname,user,password,host,port)
            cur = conn.cursor()
            cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                        'VALUES (%s, %s,%s)',
                        (name,"ppt",data)
                        )
            conn.commit()
            cur.close()
            conn.close()
    except : 
     raise 




def loadxlsfile(name,file,dbname,user,password,host,port) :
    
    
    loader = UnstructuredExcelLoader(file, mode="elements")
    data = loader.load()

    print(len(data))

    
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    #docs = text_splitter.split_documents(data)
    
      
    try:
        for d in data : 
            dat = str(d)
            print(dat)
            conn = get_db_connection(dbname,user,password,host,port)
            cur = conn.cursor()
            cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                        'VALUES (%s, %s,%s)',
                        (name,"xls",dat)
                        )
            conn.commit()
            cur.close()
            conn.close()
    except : 
     raise 







def loadpdffile(name,file,dbname,user,password,host,port) :
    
   
    loader = PyPDFLoader(file)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    
    try:
        for d in docs : 
            data = str(d)
            print(data)
            conn = get_db_connection(dbname,user,password,host,port)
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

def loadwordfile(name,file,dbname,user,password,host,port) :
    
   

    
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file)
    data = loader.load()    
  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
   
  
    
    try:
      for d in docs : 
            data = str(d)
            conn = get_db_connection(dbname,user,password,host,port)
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
 
def loadjsonfile(name,file,dbname,user,password,host,port): 
    
    with open(file,encoding="utf8") as file:
        docu = json.load(file)
        for row in docu:
            data = json.dumps(row)
            print (data)
            conn = get_db_connection(dbname,user,password,host,port)
            cur = conn.cursor()
            cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                'VALUES (%s, %s,%s)',
                (name,"json",data)
               )
            conn.commit()
            cur.close()
            conn.close()
 
def loadcsvfile(name,file,dbname,user,password,host,port) :
    
 with open(file, mode='r', encoding='utf-8-sig') as file:
  csv_reader = csv.DictReader(file)
  for row in csv_reader:
    data = json.dumps(row)
    conn = get_db_connection(dbname,user,password,host,port)
    cur = conn.cursor()
    cur.execute('INSERT INTO data (filename, typefile,chuncks)'
                'VALUES (%s, %s,%s)',
                (name,"csv",data)
               )
    conn.commit()
    cur.close()
    conn.close()
      
 
def get_completion(openai_client, model, prompt: str):    
   
    response = openai_client.chat.completions.create(
        model = model,
        messages =   prompt,
        temperature = 0.25
        
    )   
    
    return response.model_dump()

def get_db_connection(dbname,user,password,host,port):
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port)
    return conn

def authenticate(username):
    # Pour des raisons de démonstration, nous utilisons une vérification simple
    return username 
    
def generatecompletionede(openai_client,user_prompt ,username,dbname,user,password,host,port,openai_embeddings_model, openai_chat_model) -> str:
    
 
    system_prompt = '''
    You are an intelligent assistant for yourdata , please answer in the same langage use by the user . You are designed to provide helpful answers to user questions about your data.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.use the name of the file where the information is stored to provide the answer.
        - start with the hello ''' + username + '''
        - Only answer questions related to the information provided below. 
        - Write two lines of whitespace between each answer in the list.'''
        
    # system prompt

    messages = [{'role': 'system', 'content': system_prompt}]
    #user prompt
    messages.append({'role': 'user', 'content': user_prompt})
    
    vector_search_results =  ask_dbvector(user_prompt,dbname,user,password,host,port,openai_embeddings_model )
    
    for result in vector_search_results:
     
        messages.append({'role': 'system', 'content': result})
    
    response = get_completion(openai_client, openai_chat_model, messages)

    return response

def cacheresponse(user_prompt,  response , name,dbname,user,password,host,port):

    
        
    conn = get_db_connection(dbname,user,password,host,port)
    cur = conn.cursor()
    cur.execute('INSERT INTO tablecahedoc (prompt, completion, completiontokens, promptTokens,totalTokens, model,usname)'
                    'VALUES (%s, %s, %s, %s ,%s, %s,%s)',
                    (user_prompt, response['choices'][0]['message']['content'], response['usage']['completion_tokens'], response['usage']['prompt_tokens'],response['usage']['total_tokens'], response['model'] ,name))
    


    print("item inserted into cache.")
    conn.commit()
    cur.close()
    conn.close()

def cachesearch(test,name,dbname,user,password,host,port,openai_embeddings_model):
    conn = get_db_connection(dbname,user,password,host,port)
    cur = conn.cursor()
   
    query = f"""SELECT e.completion
    FROM tablecahedoc e  where e.usname = '""" + str(name) +"""' and e.dvector <=> azure_openai.create_embeddings('"""+ str(openai_embeddings_model)+"""', ' """ + str(test) + """')::vector < 0.07  ORDER BY  e.dvector <=> azure_openai.create_embeddings('"""+ str(openai_embeddings_model)+"""','""" + str(test) +"""')::vector  LIMIT 1;"""
   
    cur.execute(query)
    resutls = cur.fetchall()
   

  
    return resutls
    
def  ask_dbvector(textuser,dbname,user,password,host,port,openai_embeddings_model):
    
    conn = get_db_connection(dbname,user,password,host,port)
    cur = conn.cursor()
  
    
    query = f"""SELECT
     e.chuncks , e.filename
    FROM data e where e.dvector <=> azure_openai.create_embeddings('"""+ str(openai_embeddings_model)+"""', ' """ + str(textuser) + """')::vector < 0.25  ORDER BY  e.dvector <=> azure_openai.create_embeddings('"""+ str(openai_embeddings_model)+"""','""" + str(textuser) +"""')::vector  LIMIT 2;"""
    
    print(query)
    cur.execute(query)
    resutls = str(cur.fetchall())

                       
    chars = re.escape(string.punctuation)
    res = re.sub('['+chars+']', '',resutls)                        
                         
  
    return res


def chat_completion(openai_client,user_input,username ,dbname,user,password,host,port,openai_embeddings_model, openai_chat_model):


    # Query the chat history cache first to see if this question has been asked before
    cache_results = cachesearch(user_input ,username,dbname,user,password,host,port,openai_embeddings_model)

    if len(cache_results) > 0:
       
        return cache_results[0], True
    else:
      
   
        # Generate the completion
        completions_results = generatecompletionede(openai_client,user_input, username,dbname,user,password,host,port,openai_embeddings_model, openai_chat_model)

        # Cache the response
        cacheresponse(user_input, completions_results,username,dbname,user,password,host,port)

        
        
        return completions_results['choices'][0]['message']['content'], False


# Application Streamlit
def main():
    st.title("Connection page with Postgreqsl sample ")
    # Onglets
    
    
    dbname=config['pgdbname'],
    user=config['pguser'],
    password=config['pgpassword'],
    host=config['pghost'],
    port=config['pgport']
    st.sidebar.title("Chatbot")
    dbname =  ''.join(filter(str.isalnum, dbname))

    dbname = st.sidebar.text_input("dbname",value=dbname)
    user = ''.join(filter(str.isalnum, user))
    user = st.sidebar.text_input("pguser",value=user)
    password = ''.join(filter(str.isalnum, password))
    password = st.sidebar.text_input("pgpassword",value=password)
    host = str(host)
    host = host.replace("'","")
    host = host.replace("(","")
    host = host.replace(")","")
    host = host.replace(",","") 
    host = st.sidebar.text_input("pghost",value=host)
    port = ''.join(filter(str.isalnum, port))
    port = st.sidebar.text_input("pgport",value=port)
    openai_endpoint = config['openai_endpoint']
    openai_key = config['openai_key']
    openai_version = config['openai_version']
    openai_chat_model = config['AZURE_OPENAI_CHAT_MODEL']
    embeddingssize = config['embeddingsize']
    
    models = [
        "text-embedding-3-large",
        "text-embedding-ada-2",
            ]

    openai_embeddings_model = st.sidebar.selectbox(
        ' Chat embedding',
          (models))
    
    openai_endpoint = st.sidebar.text_input("openai_endpoint",value=openai_endpoint)
    openai_key = st.sidebar.text_input("openai_key",value=openai_key)
    openai_version = st.sidebar.text_input("openai_version",value=openai_version)   
    openai_chat_model = st.sidebar.text_input("openai_chat_model",value=openai_chat_model)
    embeddingssize = st.sidebar.text_input("embeddingssize",value=embeddingssize)
    
    global chat_history
    chat_history = []
    
    openai_client = AzureOpenAI(
     api_key = openai_key,  
     api_version = openai_version,  
     azure_endpoint =openai_endpoint 
    )
    
    
  
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
                intialize(dbname,user,password,host,port,embeddingssize,openai_embeddings_model)
                st.write("the database and collection vector and cache are created ")
                
            
            if st.button("clear the cache table "):
                st.write("start clear the Cache")
                clearcache(dbname,user,password,host,port)
                st.write("Cache cleared.")
                
           
            if st.button("delete all table to reinit"):
                st.write("delete all collection")
                cleanall(dbname,user,password,host,port)    
                st.write("all collection delete ")
            
          
  

        with tab2:
            st.header("Chargement de document ")
        
            uploaded_file = st.file_uploader("Choose your file to upload", type=["pdf", "docx","csv", "ppt","xls","xlsx" ,"pptx", "json"])
            if uploaded_file is not None:
                st.write("File selected: ", uploaded_file.name)
        
            # Enregistrer temporairement le fichier téléchargé pour obtenir le chemin absolu
                with open(uploaded_file.name, "wb") as f:
                     f.write(uploaded_file.getbuffer())

            # Obtenir le chemin absolu du fichier
                absolute_file_path = os.path.abspath(uploaded_file.name)
                st.write(f"the file is  : {absolute_file_path}")
                
                
                if st.button("load data "):
                    st.write("start the operation")
                    
                    if ".ppt" in uploaded_file.name:
                        st.write("this is a file type ppt "+ uploaded_file.name )
                        name = uploaded_file.name.replace('.ppt', '')
                        loadpptfile(name,absolute_file_path, dbname,user,password,host,port)
                        st.write("file load" +uploaded_file.name )
                    
                    elif ".doc" in uploaded_file.name:
                        st.write("this is a file type word "+ uploaded_file.name )
                        name = uploaded_file.name.replace('.doc', '')
                        loadwordfile(name,absolute_file_path, dbname,user,password,host,port)
                        st.write("file load" +uploaded_file.name )
                        
                        
                    elif ".xls" in uploaded_file.name:
                        st.write("this is a file type excel "+ uploaded_file.name )
                        name = uploaded_file.name.replace('.xls', '')
                        loadxlsfile(name,absolute_file_path, dbname,user,password,host,port)
                        st.write("file load" +uploaded_file.name )    
                  
                    elif ".pdf" in uploaded_file.name:
                        st.write("this is a file type pdf "+ uploaded_file.name )
                        name = uploaded_file.name.replace('.pdf', '')
                        loadpdffile(name,absolute_file_path ,dbname,user,password,host,port)
                        st.write("file loaded " +uploaded_file.name )
                    
                    elif ".json" in uploaded_file.name:
                        st.write("this is a file type JSON "+ uploaded_file.name )
                        name = uploaded_file.name.replace('.json', '')
                        loadjsonfile(name,uploaded_file.name,dbname,user,password,host,port)
                        st.write("file loaded " +uploaded_file.name )
                    
                    elif ".csv" in uploaded_file.name:
                        st.write("this is a file type csv "+ uploaded_file.name )
                        name = uploaded_file.name.replace('.csv', '')
                        loadcsvfile(name,uploaded_file.name,dbname,user,password,host,port)
                        st.write("file loaded " +uploaded_file.name )
                    
                    os.remove(absolute_file_path)
                    st.write(f"temp file  {absolute_file_path} was deleted .")

         
        with tab3:
            st.header("Chat")
       
                
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
                    response_payload, cached = chat_completion(openai_client,question,username,dbname,user,password,host,port,openai_embeddings_model, openai_chat_model)
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