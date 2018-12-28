#python scripts
__author__='Du Jiawei'
#Descrption:
import sys 
sys.path.append("/home/dujw/darse/Lcod")
from utils import mul_coher
import numpy as np
import os
def main():
    for i in os.listdir("."):
        print i,mul_coher(np.transpose(np.load(i)))
if __name__ == '__main__':
    main()

