import json
import time
import pandas as pd
import numpy as np
#from jtnnencoder import JTNNEmbed
# from faiss_compute import IndexFlatL2, IndexIVFFlat
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
import numpy as np
np.set_printoptions(threshold=np.inf)
import deepchem as dc

def get_data():
    df1 = pd.read_csv("../data/Smiles.csv",header=None)
    print(df1.head())
    df2 = pd.read_csv("../data/batch_0.csv",header=None)
    df3 = pd.read_csv("../data/batch_1.csv",header=None)
    df4 = pd.read_csv("../data/batch_2.csv",header=None)
    df = pd.concat([df1, df2, df3, df4])
    #df=df.drop(1,axis=1)
    print(df.head())
    #df.columns = ["smiles"]
    df.columns = ["0"]

    return df["0"].tolist()


def get_embedding(data_list):

    right_keys = []
    embeddings = []
    for i, data in enumerate(data_list):
        try:
            mols = Chem.MolFromSmiles(data)
            #jtnn = JTNNEmbed([data])
            feat3 = dc.feat.RDKitDescriptors()
            #features = jtnn.get_features()
            arr = feat3.featurize(mols)
            #print(arr.detype)
            arr.astype('float32')
            #print(arr.detype)
            embeddings.append(arr)
            right_keys.append(data)

            print(i)

        except:
           print("error: {0}, {1}".format(i, data))
    return right_keys, np.concatenate(embeddings, axis=0)


def save_dict(dict_data, save_file):
    with open(save_file, 'w') as f:
        json.dump(dict_data, f)


if __name__ == '__main__':

    t1 = time.time()
    data = get_data()
    right_keys, embeddings = get_embedding(data)
    print(len(right_keys), embeddings.shape, time.time() - t1)
    np.save('../out/embedding.npy', embeddings)

    id2name = dict(zip(range(len(right_keys)), right_keys))
    name2id = dict(zip(right_keys, range(len(right_keys))))


    # dis, ind = IndexFlatL2(embeddings)  # embeddings [512, 768],  dis:[521, 10]  index:[512,10]
    # dis: [[0, 342,4354,6..], []], ind [[1,6,5, ]]
    # print("compute cost time {0} s".format(time.time() - t1))

    # np.save('../out/dis.npy', dis)
    # np.save('../out/ind.npy', ind)

    # dis = np.load('../out/dis.npy')  # 读取保存的数据
    # ind = np.load('../out/ind.npy')  # 读取保存的数据

    # 查看每个向量索引到的最相似的向量
    # for idx, (ind_list, dis_list) in enumerate(zip(ind, dis)):
    #     print("---------{0}的最相似数据及其相似度依次为如下----------------".format(id2name[idx]))
    #     for i, d in zip(ind_list, dis_list):
    #         print("{0}， 相似度： {1}".format(id2name[i], d))

    # name2id["Clc1cccc(c1)N2[C@@H]([N@@H+]3CCC[C@@H]3C2=O)c4ccccc4"], return top10, dis ind
    # 1000  1,
