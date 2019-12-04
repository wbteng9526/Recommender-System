import numpy as np
import matplotlib.pyplot as plt
import os

path = r'C:\Users\wbten\OneDrive\Desktop\BU\Fall 2018\DS90-09\GitHub_project\RecommenderSystem\src'

os.chdir(os.path.join(path,'cf'))
from main import load_data, TEST_RATIO, get_similarity, predict
train_data_matrix, test_data_matrix = load_data(TEST_RATIO)
similarity = get_similarity(train_data_matrix, sim_type = 'cosine', cf_type = 'user')
cf_prediction = predict(train_data_matrix, similarity, cf_type = 'user')


os.chdir(os.path.join(path,'mf'))
from main import load_data, train, args
data_info = load_data(args)
mf_info = train(args, data_info)
mf_prediction = mf_info[3]

os.chdir(os.path.join(path,'nn'))
from main import load_data, train, args
data_info = load_data(args)
nn_info = train(args, data_info)
nn_prediction = nn_info[3]

def get_criterion(prediction, ground_truth, thres):

    binary_prediction = prediction >= thres
    binary_prediction = (1 * binary_prediction.reshape(1,-1)).tolist()[0]
    
    binary_real = ground_truth >= thres
    binary_real = (1 * binary_real.reshape(1,-1)).tolist()[0]

    TP = sum([int(x+y==2) for x,y in zip(binary_prediction, binary_real)])
    FP = sum([int(x-y==1) for x,y in zip(binary_prediction, binary_real)])
    TN = sum([int(x+y==0) for x,y in zip(binary_prediction, binary_real)])
    FN = sum([int(y-x==1) for x,y in zip(binary_prediction, binary_real)])
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    TPR = recall
    FPR = FP / (FP + TN)
    
    return precision, recall, TPR, FPR


def plot_criterion(prediction, ground_truth, thres_range, plot_type, color = 'red',marker = 'o'):
    
    precision_ls, recall_ls, tpr_ls, fpr_ls = [],[],[],[]
    
    for thres in thres_range:
        precision, recall, tpr, fpr = get_criterion(prediction, ground_truth, thres = thres)
        
        precision_ls.append(precision)
        recall_ls.append(recall)
        tpr_ls.append(tpr)
        fpr_ls.append(fpr)
    
    if plot_type == "PR":
        plt.plot(recall_ls,precision_ls,color = color, marker = marker)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("P-R Curve")
    else:
        plt.plot(fpr_ls, tpr_ls, color = color, marker = marker)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

   
    return None

THRES_RANGE = [0.01] + np.arange(0.05,2.6,0.05).tolist()
plt.figure()
plot_criterion(cf_prediction,test_data_matrix,plot_type = 'PR',color = 'red',marker = 'o')
plot_criterion(mf_prediction,test_data_matrix,plot_type = 'PR',color = 'green',marker = '^')
plot_criterion(nn_prediction,test_data_matrix,plot_type = 'PR',color = 'red',marker = 's')
plt.show()

plt.figure()
plot_criterion(cf_prediction,test_data_matrix,plot_type = 'ROC',color = 'red',marker = 'o')
plot_criterion(mf_prediction,test_data_matrix,plot_type = 'ROC',color = 'green',marker = '^')
plot_criterion(nn_prediction,test_data_matrix,plot_type = 'ROC',color = 'red',marker = 's')
plt.show()
