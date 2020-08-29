#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openml
import pandas as pd
import json
from pandas_profiling import ProfileReport


# In[ ]:


#get ids of active datasets
ids = [1,3,6,8,10,12,14,16,17,18,19,20,22,23,24,25,28,29,30,32,34,36,40,41,43,44,55,56,59,150,151,164,171,172,179,180,183,185,187,189,196,197,203,205,208,211,212,216,222,224,225,227,230,232,285,287,300,301,307,308,313,315,327,328,329,343,354,375,378,381,382,451,470,473,477,480,483,488,489,492,493,503,505,507,510,511,534,537,538,541,542,544,552,554,561,562,563,567,568,570,571,573,575,576,577,578,666,679,686,1027,1029,1030,1035,1036,1037,1038,1040,1042,1044,1045,1049,1050,1051,1053,1054,1056,1059,1063,1064,1065,1066,1067,1069,1070,1071,1075,1089,1090,1091,1093,1094,1095,1096,1097,1098,1099,1111,1114,1119,1168,1461,1462,1463,1464,1465,1466,1470,1471,1472,1476,1477,1478,1481,1483,1487,1489,1490,1495,1497,1499,1500,1501,1502,1503,1504,1505,1509,1510,1511,1516,1517,1518,1519,1520,1523,1524,1525,1526,1557,1567,1590,1596,1597,4135,4532,4534,4544,4548,4549,4552,4563,4675,6331]
result = openml.datasets.check_datasets_active(ids)
ds = []
for r in result:
    if result[r]:
        ds.append(r)

##extract metafeatures and store it as json        
ds_data = None
ds_meta = None

for i in range(63,len(ds)):
    ds_meta = openml.datasets.get_dataset(ds[i],download_data=True)
    print(ds_meta)
    ds_data = ds_meta.get_data()
    pd_data = pd.DataFrame(ds_data)
    pd_data.to_csv("./mf/"+str(ds[i])+'.csv',index=False)
#    profile = ProfileReport(ds_data, minimal=True, engine="python")
#    json_data = profile.to_json()
#    mf_json = json.loads(json_data)
#    with open("./mf/"+str(ds[i])+'.json', 'w') as f:
#        json.dump(mf_json, f)
#     set_mf_json.append(mf_json)
#     process_mf(mf_json)

