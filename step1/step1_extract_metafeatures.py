#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path
import pandas as pd
import json
from pandas_profiling import ProfileReport
import numpy as np
import io

input_path = "./input/"
output_path = "./output/"
def step1(database_name,data_format):
    input_step = input_path + database_name + "/"
    dirs =get_dirs(input_step)
    write_csv(dirs,data_format,database_name)

#get dirs to read files
def get_dirs(path):
    drs = os.walk(path)
    dirs = []
    for dr in drs:
        dirs.append(dr)
    return dirs
    


# In[2]:


def concat_values(json_data):
    for key in json_data.keys():
        if key == "value_counts" or key == "types":
            concat = ""
            for k in json_data[key].keys():
                if concat != "":
                    concat = concat + "|"
                concat = concat + str(k) + " " + str(json_data[key][k])
            json_data[key] = concat
    return json_data

def write_csv(dirs,file_format,database_name):
    ds_set = []
    ds_header = ["ds_id","ds_name"]
    att_cat_set = []
    att_cat_header = ["ds_id","att_name"]
    att_num_set = []
    att_num_header = ["ds_id","att_name"]

    ds_data = None
    ds_names = []

    set_ds = []
    set_at_num = []
    set_at_cat = []

    final_ds_set = []
    final_nom_set = []
    final_num_set = []
    final_ds_set_header = []
    final_nom_set_header = []
    final_num_set_header = []

    output_step = output_path + database_name + "/"
    input_step = input_path + database_name + "/"
    
    df = None
    att_list = []
    csv = []
    d = 0
    
    #read file
    for f in dirs[0][2]:
        sufix = f.split(".")[-1]
        dataset = f.split("."+sufix)[0]
        ds_names.append(dataset)
        f = input_step + f
        if file_format == "csv":
            df = pd.read_csv(f,error_bad_lines=False)
        if file_format == "excel":
            df = pd.read_excel(f)
        if file_format == "json":
            df = pd.read_json(f,orient='records')
            df = df.to_csv(sep="~",index=False)
            df = pd.read_csv(io.StringIO(df),error_bad_lines=False,sep="~")
        if len(df) > 10000:
            df = df.sample(10000)
            
        #get profile inf
        profile = ProfileReport(df, minimal=True)
        json_data = profile.to_json()
        mf_json = json.loads(json_data)

        ds_mf = mf_json["table"]
        ds_mf["name"] = ds_names[d]
        print(ds_names[d])
        ds_mf["nominal"] = {}
        ds_mf["numeric"] = {}
        set_ds.append(ds_mf)
        variables = mf_json["variables"] 
        for key in variables.keys():
            at_mf = variables[key]
            at_mf["ds_id"] = d 
            if at_mf["type"]=="Variable.TYPE_NUM":
                try:
                    del at_mf["histogram_data"]
                    del at_mf["scatter_data"]
                    del at_mf["histogram_bins"]
                except:
                    print("Something hapenned..meh")
                ds_mf["numeric"][key] = at_mf 
            else:
                ds_mf["nominal"][key] = at_mf

        att_numeric = ds_mf.pop("numeric")
        att_cat = ds_mf.pop("nominal")
    #         data = concat_values(ds_mf)
        data = ds_mf
        try:
            del data["CAT"]
            del data["BOOL"]
            del data["NUM"]
            del data["DATE"]
            del data["URL"]
            del data["COMPLEX"]
            del data["PATH"]
            del data["FILE"]
            del data["IMAGE"]
            del data["UNSUPPORTED"]
        except:
            print("...")
        data_final = {}
        id_name = data["name"].split("__")
        if len(id_name) > 1:
            data_final["ds_id"] = id_name[0]
            data_final["dataset name"] = id_name[1]
        else:
            data_final["ds_id"] = id_name[0]
            data_final["dataset name"] = id_name[0]

        ds_row = []
        final_ds_row = []

        ds_row.append(d)
        ds_row.append(data["name"])

        if len(ds_header) <= 2:
            for key in data.keys():
                ds_header.append(key)
        for key in data.keys():
            ds_row.append(data[key])
        ds_set.append(ds_row)

        ##meta-features according alserafi
        data_final["number of instances"] = data["n"]
        data_final["number of attributes"] = data["n_var"]
        data_final["dimensionality"] = float(data["n_var"]) / float(data["n"])
        num_cat = 0
        num_num = 0
        for key_types in data["types"].keys():
            if key_types == "NUM":
                num_num+=data["types"][key_types]
            else:
                print(data["types"][key_types])
                num_cat+=data["types"][key_types]
        data_final["number of nominal"] = num_cat
        data_final["number of numeric"] = num_num
        data_final["percentage of nominal"] = num_cat / float(data["n_var"])
        data_final["percentage of numeric"] = num_num / float(data["n_var"])

        #missing
        data_final["missing attribute count"] = data["n_vars_with_missing"]
        data_final["missing attribute percentage"] = float(data["n_vars_with_missing"]) / float(data["n_var"])
        num_missing_values = []
        ptg_missing_values = []

        numeric_final = {}
        #numeric
        means = []
        for key in att_numeric.keys():
            att_num_row = []
            att_num_row.append(att_numeric[key].pop("ds_id"))
            att_num_row.append(key)
    #             att_numeric[key] = concat_values(att_numeric[key])
            if len(att_num_header) <=2:
                for k in att_numeric[key].keys():
                    att_num_header.append(k)
            for k in att_numeric[key].keys():
                att_num_row.append(att_numeric[key][k])
            att_num_set.append(att_num_row)
            ####
            final_num_row = []
            means.append(att_numeric[key]["mean"])
            num_missing_values.append(att_numeric[key]["n_missing"])
            ptg_missing_values.append(att_numeric[key]["p_missing"])
            numeric_final["dataset id"] = data_final["ds_id"]
            numeric_final["attribute name"] = att_num_row[1]
            numeric_final["number distinct values"] = att_numeric[key]["distinct_count_without_nan"]
            numeric_final["percentage distinct values"] = float(att_numeric[key]["distinct_count_without_nan"]) / float(att_numeric[key]["n"])
            numeric_final["percentage missing values"] = att_numeric[key]["p_missing"]
            numeric_final["mean"] = att_numeric[key]["mean"]
            numeric_final["standard deviation"] = att_numeric[key]["std"]
            numeric_final["minimum value"] = att_numeric[key]["min"]
            numeric_final["maximum value"] = att_numeric[key]["max"]
            numeric_final["range"] = att_numeric[key]["range"]
            numeric_final["coefficient of variance"] = att_numeric[key]["cv"]
            if len(final_num_set_header) == 0:
                for final_key in numeric_final.keys():
                        final_num_set_header.append(final_key)
            for final_key in numeric_final.keys():
                final_num_row.append(numeric_final[final_key])
            final_num_set.append(final_num_row)

        if len(means) == 0:
            means = [0]
        means = np.array(means)
        data_final["average of means"] = np.average(means)
        data_final["standard deviation of means"] = np.std(means)
        data_final["minimum number of means"] = np.amin(means)
        data_final["maximum number of means"] = np.amax(means)

        #nominal    
        nominal_final = {}
        num_distinct = []  
        for key in att_cat.keys():
            if key == "<page title>":
                continue
            att_cat_row = []
            num_distinct.append(len(att_cat[key]["value_counts"].keys()))
            att_cat_row.append(att_cat[key].pop("ds_id"))
            att_cat_row.append(key)
            vcounts = []
            pvcounts = []
            string_values = ""
            for vkey in att_cat[key]["value_counts"].keys():
                vcounts.append(float(att_cat[key]["value_counts"][vkey]))
                pvcounts.append(float(att_cat[key]["value_counts"][vkey]) / float(att_cat[key]["n"]))
                if string_values != "":
                    string_values = string_values + "|"
                text = att_cat[key]["value_counts"][vkey]
                text = ''.join(char for char in text if ord(char) < 128)
                text = text.replace("~","").replace("\n"," ")
                text2 = vkey
                text2 = ''.join(char for char in text2 if ord(char) < 128)
                text2 = text2.replace("~","").replace("\n"," ")
                string_values = string_values + str(text2) + " " + str(text)

            if len(vcounts) ==0:
                vcounts = [0]
            if len(pvcounts) ==0:
                pvcounts = [0]
            vcounts = np.array(vcounts)
            pvcounts = np.array(pvcounts)
    #             att_cat[key] = concat_values(att_cat[key])
            if len(att_cat_header) <=2:
                for k in att_cat[key].keys():
                    att_cat_header.append(k)
            for k in att_cat[key].keys():
                att_cat_row.append(att_cat[key][k])
            att_cat_set.append(att_cat_row)
            #####
            final_nom_row = []
            num_missing_values.append(att_cat[key]["n_missing"])
            ptg_missing_values.append(att_cat[key]["p_missing"])
            nominal_final["dataset id"] = data_final["ds_id"]
            nominal_final["attribute name"] = att_cat_row[1]
            #############################33
    #             if ds_names[d] == "www.best-deal-items.com":
    #                 print("################THE KEY##############: "+key)
    #                 print(att_cat[key])
            #############################33333333
            nominal_final["number distinct values"] = att_cat[key]["distinct_count_without_nan"]
            nominal_final["percentage distinct values"] = float(att_cat[key]["distinct_count_without_nan"]) / float(att_cat[key]["n"])
            nominal_final["percentage missing values"] = att_cat[key]["p_missing"]
            nominal_final["mean number of string values"] = np.average(vcounts)
            nominal_final["standard deviation number of string values"] = np.std(vcounts)
            nominal_final["minimum number of string values"] = np.amin(vcounts)
            nominal_final["maximum number of string values"] = np.amax(vcounts)
            nominal_final["median percentage of string values"] = np.median(pvcounts)
            nominal_final["standard deviation percentage of string values"] = np.std(pvcounts)
            nominal_final["minimum percentage of string values"] = np.amin(pvcounts)
            nominal_final["maximum percentage of string values"] = np.amax(pvcounts)
            nominal_final["string values"] = string_values
            if len(final_nom_set_header) == 0:
                for final_key in nominal_final.keys():
                        final_nom_set_header.append(final_key)
            for final_key in nominal_final.keys():
                final_nom_row.append(nominal_final[final_key])
            final_nom_set.append(final_nom_row)

        
        data_final["average number of distintc values"] = None
        data_final["standard deviation of distintc values"] = None
        data_final["minimum number of distintc values"] = None
        data_final["maximum number of distintc values"] = None
        if len(att_cat.keys()) > 0:
            num_distinct = np.array(num_distinct)
            data_final["average number of distintc values"] = np.average(num_distinct)
            data_final["standard deviation of distintc values"] = np.std(num_distinct)
            data_final["minimum number of distintc values"] = np.amin(num_distinct)
            data_final["maximum number of distintc values"] = np.amax(num_distinct)

        ##missing
        num_missing_values = np.array(num_missing_values)
        ptg_missing_values = np.array(ptg_missing_values)
        data_final["average number of missing values"] = np.average(num_missing_values)
        data_final["standard deviation of missing values"] = np.std(num_missing_values)
        data_final["minimum number of missing values"] = np.amin(num_missing_values)
        data_final["maximum number of missing values"] = np.amax(num_missing_values)
        data_final["average number of percentage missing values"] = np.average(ptg_missing_values)
        data_final["standard deviation of percentage missing values"] = np.std(ptg_missing_values)
        data_final["minimum number of percentage missing values"] = np.amin(ptg_missing_values)
        data_final["maximum number of percentage missing values"] = np.amax(ptg_missing_values)

        if len(final_ds_set_header) ==0:
            for final_key in data_final.keys():
                final_ds_set_header.append(final_key)    
        for final_key in data_final.keys():
            final_ds_row.append(data_final[final_key])

        final_ds_set.append(final_ds_row)
        d +=1


#     output_path = input_path+"/mf_output/"    
#     output_path = "../input/monitor_mf_output/"
    if not os.path.exists(output_step):
        Path(output_step).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(final_ds_set, columns=final_ds_set_header)
    df.to_csv(output_step+"ds.csv", index=False,sep="~")
    df = pd.DataFrame(final_nom_set, columns=final_nom_set_header)
    df.to_csv(output_step+"attr_nom.csv", index=False,sep="~")
    df = pd.DataFrame(final_num_set, columns=final_num_set_header)
    df.to_csv(output_step+"attr_num.csv", index=False,sep="~")    

