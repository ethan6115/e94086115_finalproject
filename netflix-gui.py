#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import re
import random
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
import tkinter as tk
from functools import partial
import threading
 
from surprise.model_selection.validation import cross_validate

sns.set_style("darkgrid")


# In[2]:


#用來將data1到4的txt檔轉換為dataframe的格式
def readFile(file_path, rows=100000):
    data_dict = {'Cust_Id' : [], 'Movie_Id' : [], 'Rating' : [], 'Date' : []}
    f = open(file_path, "r")
    count = 0
    for line in f:
        count += 1
        if count > rows:
            break
            
        if ':' in line:
            movidId = line[:-2] # remove the last character ':'
            movieId = int(movidId)
        else:
            customerID, rating, date = line.split(',')
            data_dict['Cust_Id'].append(customerID)
            data_dict['Movie_Id'].append(movieId)
            data_dict['Rating'].append(rating)
            data_dict['Date'].append(date.rstrip("\n"))
    f.close()
            
    return pd.DataFrame(data_dict)


# In[3]:


#將data匯入到程式，因為資料及過多所以只匯入每筆資料的前100000條
df1 = readFile('./data/combined_data_1.txt', rows=100000)
df2 = readFile('./data/combined_data_2.txt', rows=100000)
df3 = readFile('./data/combined_data_3.txt', rows=100000)
df4 = readFile('./data/combined_data_4.txt', rows=100000)
df1['Rating'] = df1['Rating'].astype(float)
df2['Rating'] = df2['Rating'].astype(float)
df3['Rating'] = df3['Rating'].astype(float)
df4['Rating'] = df4['Rating'].astype(float)
df = df1.copy()
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)
#將movie_title檔案匯入
df.index = np.arange(0,len(df))
df_title = pd.read_csv('./data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])


# In[4]:


def ml_recommend():
    #不關閉程式的情況下，每次重新評分後舊評分也會留在資料中，使推薦名單更準確
    global df,df_title,df_new_list,titles
    df=df.append(df_new_list,ignore_index=True)
    
    reader = Reader()
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)
    svd = SVD()
    # Run 5-fold cross-validation and print results
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    trainset = data.build_full_trainset()
    svd.fit(trainset)
    titles = df_title.copy()

    titles['Estimate_Score'] = titles['Movie_Id'].apply(lambda x: svd.predict("user", x).est)
    titles = titles.sort_values(by=['Estimate_Score'], ascending=False)

    title10=titles.head(10).Name

    #將得出的推薦前10個電影名轉為string，用label顯示
    
    reclist=""
    for f in title10:
        reclist+=str(f)
        reclist+='\n'
        
    for widget in fm1.winfo_children():
        widget.destroy()
    #labels
    label1_4=tk.Label(fm1, text="Your recommend list:", font=('Arial',15))
    label1_4.grid(row=0, column=0, padx=20, pady=15)
    label2_3=tk.Label(fm2, text=reclist, font=('Arial',12))
    label2_3.grid(row=0, column=0, padx=20, pady=15)
    #bottom
    bt3_7=tk.Button(fm3, text='run again', font=('Arial',15),width=30,height=2,command=run)
    bt3_7.grid(row=1, column=0, padx=7, pady=10)


# In[5]:


def start(rate,ran):
    global df,df_title,count1,count2,df_new_list
    
    #清除gui frame顯示的東西
    for widget in fm2.winfo_children():
        widget.destroy()
    for widget in fm1.winfo_children():
        widget.destroy()
    #如果輸入的評分是有效的，將評分和電影放入df_new_list 
    if rate<=5:
        count1 = count1-1
        user_rate = {"Cust_Id": ["user"],			
                      "Movie_Id": [ran],
                      "Rating": [float(rate)],
                      "Date": ["2022-06-05"]}
        df_new = pd.DataFrame(user_rate)
        df_new_list=df_new_list.append(df_new,ignore_index=True)
    #若有效評分達5次，執行wait
    if count1==0:
        wait()

    elif count1>0: 
        #隨機選電影
        ran=random.randint(1,17770)
        movie_name=df_title.at[ran-1,"Name"]
        #labels
        label1_2=tk.Label(fm1, text='請以1~5分評價以下影片:', font=('Arial',15))
        label1_2.grid(row=0, column=0, padx=20, pady=15)
        label2_1=tk.Label(fm2, text=movie_name, font=('Arial',15))
        label2_1.grid(row=0, column=0)
        #bottoms
        bt3_1=tk.Button(fm3, text='1.0', font=('Arial',15),width=5,height=2,command=partial(start,1,ran))
        bt3_1.grid(row=0, column=0, padx=7, pady=10)
        bt3_2=tk.Button(fm3, text='2.0', font=('Arial',15),width=5,height=2,command=partial(start,2,ran))
        bt3_2.grid(row=0, column=1, padx=7, pady=10)
        bt3_3=tk.Button(fm3, text='3.0', font=('Arial',15),width=5,height=2,command=partial(start,3,ran))
        bt3_3.grid(row=0, column=2, padx=7, pady=10)
        bt3_4=tk.Button(fm3, text='4.0', font=('Arial',15),width=5,height=2,command=partial(start,4,ran))
        bt3_4.grid(row=0, column=3, padx=7, pady=10)
        bt3_5=tk.Button(fm3, text='5.0', font=('Arial',15),width=5,height=2,command=partial(start,5,ran))
        bt3_5.grid(row=0, column=4, padx=7, pady=10)
        bt3_6=tk.Button(fm3, text='沒看過', font=('Arial',15),width=10,height=2,command=partial(start,6,ran))
        bt3_6.grid(row=0, column=5, padx=7, pady=10)


# In[6]:


def wait():
    #clean GUI frame
    for widget in fm1.winfo_children():
        widget.destroy()
    for widget in fm2.winfo_children():
        widget.destroy()
    for widget in fm3.winfo_children():
        widget.destroy()
        
    label1_3=tk.Label(fm1, text='ml is running, please wait', font=('Arial',15))
    label1_3.grid(row=0, column=0, padx=20, pady=15)
    #用thread來同時執行GUI和machine learning 
    t = threading.Thread(target=ml_recommend)
    t.start()


# In[7]:


def run(): 
    #initial data
    global count1,count2,df_new_list,fm1,fm2,fm3,win
    count1=5
    count2=0
    df_new_list = pd.DataFrame()
    #clean GUI frame
    for widget in fm1.winfo_children():
        widget.destroy()
    for widget in fm2.winfo_children():
        widget.destroy()
    for widget in fm3.winfo_children():
        widget.destroy()
        
    #label
    label1=tk.Label(fm1, text='點選start開始評分影片', font=('Arial',15))
    label1.grid(row=0, column=0, padx=20, pady=15)

    #bottom
    bt1=tk.Button(fm2, text='start', font=('Arial',15),width=50,height=2,command=partial(start,6,0))
    bt1.grid(row=0, column=0, padx=20, pady=10)


# In[8]:


#主程式
#宣告GUI的window
win = tk.Tk()
win.title('netflix影片推薦系統')
win.geometry('600x500')
#frame
fm1=tk.Frame(win)
fm1.grid(row=0, column=0)
fm2=tk.Frame(win)
fm2.grid(row=1, column=0, columnspan=2)
fm3=tk.Frame(win)
fm3.grid(row=2, column=0)

run()
win.mainloop()


# In[ ]:




