#python scripts
__author__='Du Jiawei'
#Descrption:
import sys 
sys.path.append("/data/home/dujw/darse/Lcod")
import numpy as np
from synthetic_generator import synthetic_generate
from synthetic_generator import SyntheticProblemGenerator
global_d = 6
def eval(path):
    import pickle as pk
    sample = pk.load(open(path,"r"))
    sample = sample[0]
    pb = SyntheticProblemGenerator()
    truth = pb.get_batch_with_label(6000,shuffle=False)[1]
    acry = 0.0
    for i in range(6000):
        acry += cal_acry(sample[i],truth[i])
    print acry
    import pdb
    pdb.set_trace()

def cal_acry(pred,truth):
    tru_list = np.where(truth!=0)
    pred = np.abs(pred)
    pred_list = pred.argsort()[-1*global_d:]
    count = 0
    for i in pred_list:
        if i in tru_list[0]:
            count += 1
    return (count+0.0)/global_d
if __name__ == '__main__':
    eval("y_recon_iter_7.pkl")
