import pickle

from main import Trans_Prot, evaluate
import torch
from preprocess.loader import load_ind_data, load_model
from termcolor import colored


def predict(file,file2):
    data_iter = load_ind_data(file,file2)
    model = Trans_Prot().cuda()
    path_pretrain_model = "model_pth/Model.pt"
    model = load_model(model, path_pretrain_model)
    model.eval()
    with torch.no_grad():
        ind_performance, ind_roc_data, ind_prc_data, _, _ = evaluate(data_iter, model)
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Performance', 'red') + '=' * 16 \
                  + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        ind_performance[0], ind_performance[2], ind_performance[1], ind_performance[3],
        ind_performance[4]) + '\n' + '=' * 60

    return ind_results


file = 'D:/PIP-Trans_Prot/dataset/PIP/test.csv'
file2 = 'D:/PIP-Trans_Prot/dataset/PIP/Prot_t5_feature/prot-t5_test.csv'
ind_result = predict(file,file2)
print(ind_result)
