import tkinter as tk
from tkinter import ttk
import json
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
import numpy as np
import deepchem as dc
import time

def send_post(embedding, url="http://127.0.0.1:15000/search_embedding"):
    req = requests.post(url, json.dumps({"data": embedding}),
                        headers={"Content-Type": "application/json"})
    if req.status_code != 200:
        return req.content
    return req.json()

def load_dict(load_file):
    with open(load_file, 'r') as f:
        dict_data = json.load(f)
    return dict_data

# 在程序开始运行时加载id2name字典
id2name = load_dict("../out/id2name.json")
name2id = load_dict("../out/name2id.json")


import tkinter as tk
from tkinter import ttk
import time
from rdkit import Chem
import deepchem as dc

def search_similarities():
    # 获取当前时间戳
    query_name = entry.get() # 获取用户输入框中的Smiles字符串
    start_time = time.time()
    mols = Chem.MolFromSmiles(query_name)
    feat3 = dc.feat.RDKitDescriptors()
    arr = feat3.featurize(mols)
    one_embedding = arr
    result = send_post(one_embedding.tolist())

    if result["status"] == 1:
        ind, dis = result["data"]["ind"], result["data"]["dis"]
        # 清空文本框内容
        text.delete('1.0',tk.END)
        # 在文本框中添加查询结果
        query_result = f"---------The most similar molecules of {query_name} and their similarities are as follows----------------\n"
        for idx, (ind_list, dis_list) in enumerate(zip(ind, dis)):
            for i, d in zip(ind_list, dis_list):
                query_result += f"{id2name[str(int(i))]}, L2 Score： {d}\n"
        end_time = time.time() # 获取当前时间戳
        elapsed_time = end_time - start_time # 计算总运行时间
        query_result += f"Elapsed time for this query{elapsed_time:.2f} sceond\n" # 将查询耗时添加到字符串变量中
        text.insert(tk.END, query_result) # 将所有结果信息添加到文本框中
    else:
        # 在文本框中添加提示信息
        text.delete('1.0',tk.END)
        text.insert(1.0,"查询错误\n")

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont # 导入tkFont模块



# 创建主窗口
root = tk.Tk()
root.title("FaissMolLib")
# 创建字体对象
font = tkFont.Font(family='Helvetica', size=16)
# 添加Smiles输入框和标签，应用字体
label = ttk.Label(root,text="Please enter SMILES：", font=font)
entry = ttk.Entry(root,width=40, font=font)
label.grid(row=0,column=0,sticky="W")
entry.grid(row=1,column=0,sticky="W")

# 添加查询按钮，应用字体
search_button = tk.Button(root,text="Search",command=search_similarities, font=font)
search_button.grid(row=1,column=1,sticky="W")

# 添加文本框用于显示查询结果，应用字体
text = tk.Text(root,height=30,width=80, font=font)
text.grid(row=2,column=0,columnspan=2,sticky="NESW")

# 让文本框随着窗口大小变化而变化
root.columnconfigure(0, weight=1)
root.rowconfigure(2, weight=1)
text.configure(wrap="word")

root.mainloop()