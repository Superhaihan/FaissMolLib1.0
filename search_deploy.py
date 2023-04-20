import time
import json
import faiss
import time
import requests
import numpy as np
from flask import Flask, request
#from jtnnencoder import JTNNEmbed
from tqdm import tqdm
import numpy as np

#from rdkit import Chem
#from rdkit.Chem import AllChem, Draw
#from rdkit.Chem import MACCSkeys
import numpy as np
#np.set_printoptions(threshold=np.inf)
import deepchem as dc
app = Flask(__name__)

embeddings = np.load('../out/embedding.npy')
print("embeddings shape {0}".format(embeddings.shape))

t1 = time.time()
# 查找 TopK 的最相似向量, 修改这里
top_k = 15
nums_user, dimension = embeddings.shape[0], embeddings.shape[1]
embeddings=embeddings.astype('float32')
cpu_index = faiss.IndexFlatL2(dimension)  # 构建索引index
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.add(embeddings)  # 添加数据

print("train cost {0}".format(time.time() - t1))


@app.route('/search_embedding', methods=['POST'])
def search():
    if request.method == 'POST':
        data = request.json.get("data")
    else:
        return {"data": {}, "status": 0, "message": "request is not post type."}

    data = np.array(data).astype('float32')
    dis, ind = gpu_index.search(data, top_k)

    return {"data": {"dis": dis.tolist(), "ind": ind.tolist()}, "status": 1, "message": "succeed"}

def load_dict(load_file):
    with open(load_file, 'r') as f:
        dict_data = json.load(f)
    return dict_data

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=15000)