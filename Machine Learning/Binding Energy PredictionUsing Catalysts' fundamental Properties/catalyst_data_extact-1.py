import json
import pandas as pd

with open('catalyst_data') as f:
  content = f.read()

l = json.loads(content)
data = l['data']['reactions']['edges']
refined_data = []
for x in data:
  refined_data.append(x['node'])

df = pd.DataFrame(refined_data)
df.to_csv('catalyst_data.csv')