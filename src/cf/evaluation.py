import numpy as np
import matplotlib.pyplot as plt
from model import predict, top_k_similarity, weighted_similarity,rmse

def visualization(ratings, ground_truth, f, range_list):
    user_test_rmse_ls = []
    params = []
    for param in range_list:
        if f == top_k_similarity:
            user_similarity = f(ratings,k=param)
            xlabel = "K"
        elif f == weighted_similarity:
            user_similarity = f(ratings,sim_type = 'correlation',alpha=param)
            xlabel = "alpha"
            
        user_prediction = predict(ratings, user_similarity, cf_type='user')
        user_test_rmse = rmse(user_prediction, ground_truth)
        user_test_rmse_ls.append(user_test_rmse)
        params.append(param)
        
    plt.plot(params,user_test_rmse_ls)
    
    plt.xlabel(xlabel)
    plt.ylabel("User RMSE")
    
    return np.argmin(user_test_rmse_ls)


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


def plot_criterion(prediction, ground_truth, thres_range, plot_type):
    
    precision_ls, recall_ls, tpr_ls, fpr_ls = [],[],[],[]
    
    for thres in thres_range:
        precision, recall, tpr, fpr = get_criterion(prediction, ground_truth, thres = thres)
        
        precision_ls.append(precision)
        recall_ls.append(recall)
        tpr_ls.append(tpr)
        fpr_ls.append(fpr)
    
    if plot_type == "PR":
        plt.plot(recall_ls,precision_ls)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("P-R Curve")
    else:
        plt.plot(fpr_ls, tpr_ls)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

   
    return None
    
    
    

