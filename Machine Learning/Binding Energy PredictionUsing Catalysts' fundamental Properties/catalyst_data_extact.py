import requests
import pprint
import sys
import string
import json
import io
import copy
import csv

import ase.io
import ase.calculators.singlepoint

GRAPHQL = 'http://api.catalysis-hub.org/graphql'

def fetch(query):
    return requests.get(
        GRAPHQL, {'query': query}
    ).json()['data']

def reactions_from_dataset(pub_id, page_size=10):
    reactions = []
    has_next_page = True
    start_cursor = ''
    page = 0
    while has_next_page:
        data = fetch("""{{
      reactions(pubId: "{pub_id}", first: {page_size}, after: "{start_cursor}") {{
        totalCount
        pageInfo {{
          hasNextPage
          hasPreviousPage
          startCursor
          endCursor 
        }}  
        edges {{
          node {{
            Equation
            reactants
            products
            reactionEnergy
            reactionSystems {{
              name
              systems {{
                energy
                InputFile(format: "json")
              }}
            }}  
          }}  
        }}  
      }}    
    }}""".format(start_cursor=start_cursor,
                 page_size=page_size,
                 pub_id=pub_id,
                ))
        has_next_page = data['reactions']['pageInfo']['hasNextPage']
        start_cursor = data['reactions']['pageInfo']['endCursor']
        page += 1
        print(has_next_page, start_cursor, page_size * page, data['reactions']['totalCount'])
        reactions.extend(map(lambda x: x['node'], data['reactions']['edges']))
        
    return reactions

raw_reactions = reactions_from_dataset("MamunHighT2019", page_size=100)

def aseify_reactions(reactions):
    for i, reaction in enumerate(reactions):
        for j, _ in enumerate(reactions[i]['reactionSystems']):
            with io.StringIO() as tmp_file:
                system = reactions[i]['reactionSystems'][j].pop('systems')
                tmp_file.write(system.pop('InputFile'))
                tmp_file.seek(0)
                atoms = ase.io.read(tmp_file, format='json')
            calculator = ase.calculators.singlepoint.SinglePointCalculator(
                atoms,
                energy=system.pop('energy')
            )
            atoms.set_calculator(calculator)
            #print(atoms.get_potential_energy())
            reactions[i]['reactionSystems'][j]['atoms'] = atoms
        # flatten list further into {name: atoms, ...} dictionary
        reactions[i]['reactionSystems'] = {x['name']: x['atoms']
                                          for x in reactions[i]['reactionSystems']}
        
reactions = copy.deepcopy(raw_reactions)
aseify_reactions(reactions)

keys = reactions[0].keys()
with open('catalysis_data.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(reactions)
