import os
from functions import get_dataframe, preprocess_dataframe


dirname = os.path.dirname(__file__)
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\LDA_final'
output_csv_file = os.path.join(dirname, 'labelled_dataset.csv')
file_test = os.path.join(dirname,'New Clauses_ SLB Examples for Quickstudies_20190301_143554_182.xls')
df = get_dataframe(file_test, sheet='New Clauses  SLB Examples for Q')
df.rename(columns={'Result': 'Text', 'Field Name': 'Clause'}, inplace=True)
df = preprocess_dataframe(df, .85)
df.to_csv (output_csv_file,  mode='a',index = False)