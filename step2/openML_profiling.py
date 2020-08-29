import openml
import pandas as pd
import json
from pandas_profiling import ProfileReport

##extract metafeatures and store it as json 
ids = [1,3,6,8,10,12,14,16,17,18,19,20,22,23,24,25,28,29,30,32,34,36,40,41,43,44,55,56,59,150,151,164,171,172,179,180,183,185,187,189,196,197,203,205,208,211,212,216,222,224,225,227,230,232,285,287,300,301,307,308,313,315,327,328,329,343,354,375,378,381,382,451,470,473,477,480,483,488,489,492,493,503,505,507,510,511,534,537,538,541,542,544,552,554,561,562,563,567,568,570,571,573,575,576,577,578,666,679,686,1027,1029,1030,1035,1036,1037,1038,1040,1042,1044,1045,1049,1050,1051,1053,1054,1056,1059,1063,1064,1065,1066,1067,1069,1070,1071,1075,1089,1090,1091,1093,1094,1095,1096,1097,1098,1099,1111,1114,1119,1168,1461,1462,1463,1464,1465,1466,1470,1471,1472,1476,1477,1478,1481,1483,1487,1489,1490,1495,1497,1499,1500,1501,1502,1503,1504,1505,1509,1510,1511,1516,1517,1518,1519,1520,1523,1524,1525,1526,1557,1567,1590,1596,1597,4135,4532,4534,4544,4548,4549,4552,4563,4675,6331]

ds_data = None
ds_meta = None
ds_names = []

for i in range(len(ids)):
    ds_meta = openml.datasets.get_dataset(ids[i],download_data=False)
    ds_names.append(ds_meta.name)
    
    
set_ds = []
set_at_num = []
set_at_cat = []
for i in range(141,len(ids)):
    print("working with: "+str(ids[i]))
    df = pd.read_csv("./mf/"+str(ids[i])+".csv", engine="python")
    if len(df) > 10000:
        df = df.sample(10000)
    profile = ProfileReport(df, minimal=True)
    json_data = profile.to_json()
    mf_json = json.loads(json_data)
    ds_mf = mf_json["table"]
    ds_mf["name"] = ds_names[i]
    ds_mf["nominal"] = {}
    ds_mf["numeric"] = {}
    set_ds.append(ds_mf)
    variables = mf_json["variables"]  
    for key in variables.keys():
        at_mf = variables[key]
        at_mf["ds_id"] = ids[i] 
        if at_mf["type"]=="Variable.TYPE_NUM":
            del at_mf["histogram_data"]
            del at_mf["scatter_data"]
            del at_mf["histogram_bins"]
            ds_mf["numeric"][key] = at_mf 
        else:
            ds_mf["nominal"][key] = at_mf 
    with open("./mf_extracted/"+str(ids[i])+'.json', 'w') as f:
        json.dump(ds_mf, f)
