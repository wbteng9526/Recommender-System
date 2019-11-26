import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()]
    ground_truth = ground_truth[ground_truth.nonzero()]
    return np.sqrt(mean_squared_error(prediction, ground_truth))


def predict(ratings, similarity,cf_type = 'user',centered = True):
    # User-CF

    if cf_type == 'user':
        
        if centered == True:
            # average bias
            mean_user_rating = ratings.mean(axis=1, keepdims=True)
            ratings_diff = (ratings - mean_user_rating)
            pred = mean_user_rating + similarity.dot(ratings_diff) / similarity.sum(axis=1, keepdims=True)
        
        else:
            # average score
            pred = similarity.dot(ratings) / similarity.sum(axis=1, keepdims=True)
    # Item-CF
    elif cf_type == 'item':
        pred = ratings.dot(similarity) / similarity.sum(axis=1)
    else:
        return None
    return pred


def top_k_similarity(ratings, k, sim_type = 'cosine', cf_type = 'user'):
    similarity = get_similarity(ratings, sim_type = sim_type, cf_type = cf_type)
    length = similarity.shape[0]
    output1 = similarity.copy()
    output2 = similarity.copy()
    for i in range(length):
        user_array = output1[i]
        user_array_sorted = selection_sort(user_array)
        thres = user_array_sorted[length - k]
        for j in range(length):
            if output2[i,j] < thres:
                output2[i,j] = 0
    return output2
    


def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x


def weighted_similarity(ratings, sim_type = 'cosine', cf_type = 'user',alpha=50):
    similarity = get_similarity(ratings, sim_type = sim_type, cf_type = cf_type)
    nrow= ratings.shape[0]
    user_weight_matrix = np.zeros(shape = (nrow, nrow))
    for i in range(nrow):
        for j in range((i+1),nrow):
            user_weight_matrix[i,j] = user_weight_matrix[j,i] = get_weight(ratings[i],ratings[j],alpha = alpha)
    
    return similarity * user_weight_matrix


def get_weight(arr1, arr2, alpha=50):
    arr_mul = arr1 * arr2
    return min(np.count_nonzero(arr_mul)/alpha,1)



def get_similarity(ratings, sim_type = 'cosine', cf_type = 'user'):
    
    if cf_type == 'user':
        return 1 - pairwise_distances(ratings, metric = sim_type)
    else:
        return 1 - pairwise_distances(ratings.T, metric = sim_type)





