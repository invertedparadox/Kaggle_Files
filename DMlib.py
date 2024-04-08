import pandas as pd
import os
from PIL import Image

# SOURCE: Demetrius Gulewicz
def get_categories_train(face_filt_dir, category_file, train_file):
    # get pre processed data
    valid_train = pd.read_csv(face_filt_dir, delimiter =',', names=['Key', 'q', 'ar', 'px', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
    valid_ids = valid_train['Key']

    # get output data
    df_unique = pd.read_csv(category_file)
    df_all = pd.read_csv(train_file)

    # remove bad ids
    df_all_filt = df_all[df_all['Key'].isin(valid_ids)]

    # get initial situation for categories
    dict_all_arr = {}
    dict_all_num = {}
    for target_name in df_unique["Value"]:
        target_cols = df_all[df_all["Value"] == target_name]
        dict_all_arr[target_name] = target_cols["Key"]
        dict_all_num[target_name] = target_cols["Key"].count()

    df_all_num = pd.DataFrame({"Name": dict_all_num.keys(), "Count": dict_all_num.values()})
    print([df_all_num["Count"].max(), df_all_num["Count"].min(), df_all_num["Count"].sum()])

    # get final situation for categories
    dict_all_filt_arr = {}
    dict_all_filt_num = {}
    for target_name in df_unique["Value"]:
        target_cols = df_all_filt[df_all_filt["Value"] == target_name]
        dict_all_filt_arr[target_name] = target_cols["Key"]
        dict_all_filt_num[target_name] = target_cols["Key"].count()

    df_all_filt_num = pd.DataFrame({"Name": dict_all_filt_num.keys(), "Count": dict_all_filt_num.values()})
    print([df_all_filt_num["Count"].max(), df_all_filt_num["Count"].min(), df_all_filt_num["Count"].sum()])
    
    return df_unique, dict_all_filt_arr

# SOURCE: Demetrius Gulewicz
def organize_files_train(df_unique, dict_all_filt_arr, NN_train_dir, NN_validate_dir, clean_train_dir, r_tv):
    num_catergories = df_unique['Key'].count()
    names = df_unique['Value'].values

    for i in range(num_catergories):
        # create folder
        train_path = NN_train_dir + names[i]
        validate_path = NN_validate_dir + names[i]
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(validate_path, exist_ok=True)
        
        idxs = dict_all_filt_arr[names[i]].index
        num_idxs = len(idxs)
        num_train = int(r_tv*num_idxs)
        
        training_set = idxs[0:num_train]
        validate_set = idxs[num_train:]
        
        # transfer all training images to training folder
        for i in training_set:
            img = Image.open(clean_train_dir + str(i) + '.jpg')
            img.save(train_path + '/' + str(i) + '.jpg')
            img.close()
            
        # transfer all validation images to validation folder
        for i in validate_set:
            img = Image.open(clean_train_dir + str(i) + '.jpg')
            img.save(validate_path + '/' + str(i) + '.jpg')
            img.close()