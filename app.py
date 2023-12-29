import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
import requests
from langchain.chat_models import ChatOpenAI
from PIL import Image
from io import BytesIO
from tools import *
from flask import url_for
from flask import send_from_directory
from flask import jsonify, request, url_for
os.environ["OPENAI_API_KEY"] = 'PUT APT KEY HERE'


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm=ChatOpenAI(temperature=0.1, model_name='gpt-4')
tools = load_tools([], llm=llm)
tools.append(Tool(name="query-to-SMILES", func=query_to_smiles,
                  description="SHOULD BE USED BEFORE ANY PREDICTIONS useful to convert chemical name (IUPAC) to SMILES", ))
tools.append(Tool(name="reaction_prediction", func=reaction_predict_bulk,
                  description="REQUIRES VALID SMILES from queries predicts the products of chemical reactions. input: String of smiles joined by .", ))
tools.append(Tool(name="SMILES-to-query", func=smiles_to_query,
                  description="useful to convert chemical SMILES to name (IUPAC)", ))
tools.append(Tool(name="Retrosynthesis", func=retrosynthesis,
                  description="creates retrosynthesis from SMILEs", ))
tools.append(Tool(name="synthetic-procedure", func=synthesis_procedure,
                  description="creates synthetic procedures from SMILEs", ))
'''tools.append(Tool(name="ir-prediction", func=ir_prediction,
                  description="Predicts IR spectra from smiles string", ))'''
tools.append(Tool(name="draw-molecule", func=draw_molecule,
                  description="Draws a molecule from there smiles string", ))

agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# Create a Flask application instance
app = Flask(__name__)
# Generate a random secret key for the Flask application
app.secret_key = os.urandom(24)

# Define the route for the root URL, rendering the index.html template
@app.route("/")
def index():
  return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
  data = request.get_json()
  user_message = data.get("message")
  
  reply = agent_chain.run(input=user_message)
  response = {'message': reply, 'image': False}
  
  if MyConstants.image:
    image_url = url_for('static', filename=MyConstants.image)
    response['image'] = image_url
    MyConstants.image = False
  
  return jsonify(response)

# Start the Flask application, listening on all interfaces and port 8080
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
