from rxn4chemistry import RXN4ChemistryWrapper
import time
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from IPython.display import display
from rdkit.Chem import Draw
import json
import matplotlib.pyplot as plt
import torch
import warnings
#from dataset import create_pytorch_geometric_graph_data_list_from_smiles_and_labels
#from model import GCN
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.signal import find_peaks
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
os.environ["OPENAI_API_KEY"] = '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'

class MyConstants:
    sequence_id = None
    image = False


#function for IR prediction using previously trained model
'''def ir_prediction(smiles):
    graph_data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels([smiles], [1])
    model = GCN()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    with torch.no_grad():
        predicted_spectrum = list(model(graph_data[0].x, graph_data[0].edge_index, None, graph_data[0].edge_weight)[0][0])
    inverted_y_values = [1.0 - y for y in predicted_spectrum]
    plt.plot(range(0, 1000), inverted_y_values, color='red')
    plt.title('Predicted IR Spectrum of ' + smiles)
    plt.xlabel('Wavenumber (cm-1)')
    plt.ylabel('Transmittance')
    plt.xlim(0, 1000)
    plt.xticks([0, 200, 400, 600, 800, 1000], ['4000', '3000', '2000', '1000', '500', '400'])
    MyConstants.image = 'plot.png'
    plt.savefig('static/plot.png')
    peaks, _ = find_peaks(predicted_spectrum, height=0.24)
    peak_wavenumbers = []
    for peak_index in peaks:
        peak_wavenumbers.append(round(4000 - (3.6 * peak_index)))
    main_peaks = 'The main peaks in the spectrum are '
    for wavenumber in peak_wavenumbers:
        main_peaks += str(wavenumber) + ', '
    main_peaks += 'cm-1'
    return main_peaks'''

def retrosynthesis(product_smiles):
    while True:
        try:
            api_key = "PUT API KEY HERE"
            rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
            rxn4chemistry_wrapper.set_project('SET PROJECT ID HERE')
            response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(product=product_smiles)
            time.sleep(15)
            while True:
                results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(response['prediction_id'])
                print(results['status'])
                if results['status'] == 'SUCCESS':
                    break
                else:
                    time.sleep(25)
            outcome_molecules = results['response']['payload']['sequences'][0]['outcomeMolecules']
            reagents = 'Reagents: \n'
            for molecule in outcome_molecules:
                reagents += molecule['moleculeName'] + '\n'
            for index, path in enumerate(results['retrosynthetic_paths']):
                if index == 0:
                    confidence = path['confidence']
                    reaction_class = path['rclass']
                    for i, reaction in enumerate(collect_reactions(path)):
                        print('Saving path {} with confidence {} to file:'.format(index, path['confidence']))
                        reaction_smiles = collect_reactions_smiles(path)
                        MyConstants.image = 'retro_path.png'
                        image = Chem.Draw.ReactionToImage(reaction)
                        image.save('static/retro_path.png')
                    else:
                        break
                else:
                    break
            path = results['retrosynthetic_paths'][0]
            MyConstants.sequence_id = path['sequenceId']
            retro_synthesis_info = 'Reaction Smiles: ' + reaction_smiles[0] + '\n' + 'Reaction class: ' + reaction_class + '\n' + reagents + '\n' + 'Confidence: ' + str(confidence) + '\n\nRetroSynthesis Complete'
            print(retro_synthesis_info)
            chat_model = ChatOpenAI(temperature=0.1, model_name='gpt-4')
            output = chat_model([HumanMessage(content='You must write up this retrosynthesis and include all the information given but only the information given and nothing else. use the reaction smiles to help you: \n' + retro_synthesis_info)])
            return output
        except:
            return 'ERROR: Invalid SMILES string'

def collect_reactions(tree):
    reactions = []
    reactionssmiles=[]
    if 'children' in tree and len(tree['children']):
        reactionssmiles.append('{}>>{}'.format('.'.join([node['smiles'] for node in tree['children']]),tree['smiles']))
        reactions.append(
            AllChem.ReactionFromSmarts('{}>>{}'.format(
                '.'.join([node['smiles'] for node in tree['children']]),
                tree['smiles']
            ), useSmiles=True)
        )
    for node in tree['children']:
        reactions.extend(collect_reactions(node))
    return reactions

def collect_reactions_smiles(tree):
    reactionssmiles = []
    if 'children' in tree and len(tree['children']):
        reactionssmiles.append('{}>>{}'.format('.'.join([node['smiles'] for node in tree['children']]), tree['smiles']))
    return reactionssmiles

def reaction_predict_bulk(precursors_list):
    while True:
        try:
            api_key = "SET API KEY HERE"
            rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
            response = rxn4chemistry_wrapper.predict_reaction_batch(precursors_list=[precursors_list])
            time.sleep(3)
            results = (rxn4chemistry_wrapper.get_predict_reaction_batch_results(response["task_id"]))
            output = []
            output = output + [(results['predictions'][0]['smiles']), (results['predictions'][0]['confidence'])]
            return str(output[0].split('>>')[1]), str(output[1])[0:6]
        except:
            try:
                if results['response']['error'] == 'Too Many Requests':
                    time.sleep(80)
                else:
                    return 'Invalid reactants string'
            except:
                return 'Invalid input format e.g "CCN.CCO"'

def query_to_smiles(query):
  try:
    # Querying pubchem
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/CanonicalSMILES/JSON"
    response = requests.get(url)
    result = response.json()
    # Checking if pubchem returned any smiles
    return result["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
  except:
    return 'Invalid IUPAC must input one at a time'
  
def smiles_to_query(smiles):
  try:
    # Querying pubchem
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
    response = requests.get(url)
    result = response.json()
    # Checking if pubchem returned any IUPAC names
    return str(result["PropertyTable"]["Properties"][0]["IUPACName"])
  except:
    return 'Invalid SMILES input one at a time'
  
def synthesis_procedure(smiles):
    api_key = "SET API KEY HERE"
    rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
    rxn4chemistry_wrapper.set_project('SET PROJECT ID HERE')
    response = rxn4chemistry_wrapper.create_synthesis_from_sequence(sequence_id=MyConstants.sequence_id)
    overall=(response['response']['payload']['sequences'][0]['tree']['productMassAndReactionInformation'])
    procedure = 'Product amount: '+ str(overall['quantity'])+ ' ' + str(overall['unit']) + ', Predicted Product yield:' + str(overall['reactionYield'])+ '\nSynthetic procedure: '
    data=(response['response']['payload']['sequences'][0]['tree']['initialActions'])
    print(data)
    for step in data:
        if step['name'] == 'add':
            atmosphere = step['content']['atmosphere']
            material = step['content']['material']['value']
            quantity = str(step['content']['material']['quantity']['value'])
            unit = str(step['content']['material']['quantity']['unit'])
            dropwise = step['content']['dropwise']['value']
            procedure = procedure + '\n'+ (f"Add {quantity} {unit} of {material} {'dropwise' if dropwise else ''} ")
            if atmosphere != None:
                procedure = procedure + (f"in {atmosphere} atmosphere")
        elif step['name'] == 'set-temperature':
            temperature = str(step['content']['temperature']['value'])
            procedure = procedure + '\n'+(f"Set temperature to {temperature}°C")
        elif step['name'] == 'reflux':
            duration = str(step['content']['duration']['value'])
            unit = (step['content']['duration']['unit'])
            procedure = procedure + '\n'+(f"Reflux for {duration} {unit}")
        elif step['name'] == 'stir':
            atmosphere = step['content']['atmosphere']
            duration = str(step['content']['duration']['value'])
            unit = step['content']['duration']['unit']
            temperature = str(step['content']['temperature']['value'])
            unit_temp = step['content']['temperature']['unit']
            #stirspped = step['content']['stirringSpeed']['value']
            procedure = procedure + '\n'+(f"Stir for {duration} {unit} at {temperature}°{unit_temp}")
            if atmosphere != None:
                procedure = procedure + (f"in {atmosphere} atmosphere")
        elif step['name'] == 'purify':
            procedure = procedure + '\n' + ("Purfify")
        elif step['name'] == 'concentrate':
            procedure = procedure + '\n' + ("Concentrate")
    MyConstants.sequence_id = None
    llm = ChatOpenAI(temperature=0.3, model_name='gpt-4')
    output = llm(([HumanMessage(content='You must write the given expirimental procedure up adding equipment and making it flow \n' + procedure + '\n')]))
    return output

def draw_molecule(smiles):
    # Create an RDKit molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # Generate a 2D depiction of the molecule
    img = Draw.MolToImage(mol)
    MyConstants.image='molecule.png'
    # Save the image to a file
    img.save("static/molecule.png")
    return (smiles + ' Has been drawn')
