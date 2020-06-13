import pandas as pd
import numpy as np
import os
import sys
from sklearn import model_selection
from PIL import Image

#creating folds

df["kfold"] = -1
df = df.sample(frac = 1).reset_index(drop = True)
y = df.target.values
kf = model_selection.StratifiedKFold(n_splits=518)
for fold_,(_,test_idx) in enumerate(kf.split(X=df,y=y)):
    df.loc[test_idx,"kfold"] = fold_

#data generator

def fold_datagenerator(number_of_folds,fold_size):
    while True:
        start = 0
        end = fold_size
        while start  < number_of_folds: 
            x = df[df["kfold"]==start].image_name.values
            images_list=[]
            for i in x:
                image = Image.open(df_path + i).resize((512,512))
                images_list.append(np.asarray(image)/255.0)
            y = df[df["kfold"]==start].target.values
            yield np.array(images_list),np.array(y)
            
            start += fold_size
            end += fold_size