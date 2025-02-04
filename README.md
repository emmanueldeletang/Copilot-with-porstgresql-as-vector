# Copilot-with-porstgresql-as-vector
a full web application where you can load file( csv , json , word , pdf ) and make llm and vector search with sample code to bulid your copilot and load your data inside and put question and answer in cache to save time


## Features
- Vector search using Azure postgresql
- Create embeddings with the extension openai inside Postgresql 
- Use database as cache to save latency

## Requirements
- Tested only with Python 3.11
- Azure OpenAI account
- Azure postgresql

## Setup
- Create virtual environment: python -m venv .venv
- Activate virtual ennvironment: .venv\scripts\activate
- Install required libraries: pip install -r requirements.txt
- Replace keys with your own values in example.env
- configure inside you postgresql the correct extension and activate as show in the file postgresqlinit( extension , link to openai model ... ) i use diskann as extension too , if not enable for you use hnsw and replace in the code 
- don't forget to have the model openAI one text-embbeding and one GPT ( can be 4.0 ,3.5 ) ... and configure all in the example.env , with the size of the vector 
- TO LAUNCH THE APPLICATION JUST do Streamlit run pgtest.py
- enter a login name and after in the first login and all the collections will be create just push the button ( create vector db  )
- have fun
- some sample document are in the dataset folder

