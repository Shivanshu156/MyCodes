#!/usr/bin/env python
# coding: utf-8

# In[3]:


file_dict = [    {'name':"ADNOC - UAE - Stimulation Services_TEXT_Refined.txt",
'page': [1, 2, 3,6,7,8,9,10,11,12,73,74,78,79,80,81,83,84,85,86,87,88,89,91,98,100,99,101,105,106,108,109,110,111,
             112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138
             ,139,142,144,145,146,147,148,149,150,151,152,153,154,156,158,159,162,163,167,169,170,171,172,173,174,175,
             176,177,178,179,184,185,188,192,199,204,206,205, 318,319,320,321,322,323,324,347,348,356,357,358,
             370,371,373,383,384,385,386,387,389,390,396,398,399,402,407,408,409,410,412,413,428,504,505,510,513,514
             ,530,529,694,737,738],
'page_range' : {206:313, 515:526,532:545,575:580,648:651,660:692,725:734,739:744,746:750,757:799,821:831}
     },
        {'name':"AGR Norway - Drilling and Well Services_T&C only_No Price Adjustment_TEXT_Refined.txt",
'page': [],
'page_range' : {1:3,6:8}
     },
      {'name':"Apache North Sea - ESP_TEXT_Refined.txt",
'page': [],
'page_range' : {15:19}
     },
             {'name':"Arrow Energy - Cementing Contract_TEXT_Refined.txt",
'page': [1,2,155],
'page_range' : {5:8, 84:86,101:104, 110:113, 116:122,134:136,146:148,158:163,178:193,226:249,255:266,285:288   }
     },
             {'name':"Australian Nickel - ESP Equipment and Services_TEXT_Refined.txt",
'page': [],
'page_range' : {27:30, 36:39}
     },      
             {'name':"B2071_Anton Oilfield Services (Group) Ltd._CG-AT-2012-1152_Contract_TEXT_Refined.txt",
'page': [23],
'page_range' : {}
     },
                  {'name':"BG - India - WL_TEXT_Refined.txt",
'page': [102,103,111,112],
'page_range' : {62:66, 82:88,114:120 }
     },
                  {'name':"BHP Billiton - S Africa - Geophysical Services_TEXT_Refined.txt",
'page': [],
'page_range' : {}
     },
                  {'name':"BHP Commercial - Mud Logging_TEXT_Refined.txt",
'page': [54],
'page_range' : {31:37, }
     },
                  {'name':"BMS - Algeria - Completions Services_TEXT_Refined.txt",
'page': [23,44],
'page_range' : {51:53,58:59,65:144,171:173 }
     }]


# In[4]:


import json
open('bad_pages.json', 'w').write(json.dumps(file_dict))


# In[ ]:




