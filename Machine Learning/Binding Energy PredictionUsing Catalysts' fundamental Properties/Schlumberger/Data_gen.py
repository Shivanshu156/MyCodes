from pprint import pprint
import pandas as pd
import json
import os
from glob import glob

text_file_dirname = r'C:\Users\sverma9\Desktop\Shivanshu\Contracts\Text files'
output_csv_file = r'C:\Users\sverma9\Desktop\Shivanshu\complete_dataset_optimised.csv'

file_add_list = glob(os.path.join(text_file_dirname,'*.txt'))
file_names = list(map(lambda x : os.path.basename(x), file_add_list))

texts = []
for j in range(0,len(file_add_list)):
    file_add = file_add_list[j]
    file_name = os.path.basename(file_add)
    text=[]
    with open(file_add) as f:
        contents = f.read()
    pages = contents.split('\n\nTHIS IS A PAGE BREAK\n\n') 
    print('Number of pages in file ',file_name,' are ',len(pages))
    
    for i in range(0, len(pages)):
        paragraphs = pages[i].split('\n')
        text.append(paragraphs)
        
    texts.append(text)   


df = pd.DataFrame(columns=['Contract Name', 'Page_no', 'Paragraph'])

for i in range(len(texts)):
    df1 = pd.DataFrame({'Contract Name': file_names[i], 'Page_no':
                       list(range(len(texts[i]))),'Paragraph': texts[i] })
    df = df.append(df1,ignore_index=True)


rows = []
_ = df.apply(lambda row: [rows.append([row['Contract Name'],row['Page_no'], nn])
                         for nn in row.Paragraph], axis=1)
df_new = pd.DataFrame(rows, columns=df.columns)

df_new
df_new = df_new[df_new.Paragraph != '']
df_new = df_new[df_new.Paragraph != '\n']

df_new.to_csv (output_csv_file,  mode='a',index = False)

