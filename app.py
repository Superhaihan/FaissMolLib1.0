import json
import numpy as np
import requests
import time
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
import deepchem as dc
from flask import Flask, render_template, request

def send_post(embedding,url="http://127.0.0.1:15000/search_embedding"):
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

app = Flask(__name__)

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

@app.route('/', methods=['GET', 'POST'])
def search_similarities():
    if request.method == 'POST':
        query_name = request.form['query_name']  # 获取用户输入框中的Smiles字符串

        if is_valid_smiles(query_name):
            start_time = time.time()
            mols = Chem.MolFromSmiles(query_name)
            feat3 = dc.feat.RDKitDescriptors()
            arr = feat3.featurize(mols)
            one_embedding = arr
            result = send_post(one_embedding.tolist())

            if result["status"] == 1:
                ind, dis = result["data"]["ind"], result["data"]["dis"]
                # 在文本框中添加查询结果
                query_result = f"<p>---------The most similar molecules of {query_name} and their similarities are as follows----------------</p>"
                for idx, (ind_list, dis_list) in enumerate(zip(ind, dis)):
                    for i, d in zip(ind_list, dis_list):
                        query_result += f"<p>{id2name[str(int(i))]}, L2 Score: {d}</p>"
                end_time = time.time() # 获取当前时间戳
                elapsed_time = end_time - start_time # 计算总运行时间
                query_result += f"<p>Elapsed time for this query: {elapsed_time:.2f} seconds</p>" # 将查询耗时添加到字符串变量中
            else:
                # 在文本框中添加提示信息
                query_result = "<p>查询错误</p>"
            return render_template('index.html', query_result=query_result)
        else:
            error_message = "Invalid SMILES format."
            return render_template('index.html', error_message=error_message)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=True)
