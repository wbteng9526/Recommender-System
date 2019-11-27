## Recommender System

### Introduction
This repository is created for comparing the 3 models for building up a recommender system:
- Collaborative filtering
- Matrix factorization
- Neural Network

### Methodology

- Collaborative Filtering
  - Similarity
    - Cosine
    
    ![github-small](img/cfsimcos.png)
    - Pearson coefficient
    
    ![github-small](img/cfsimpearson.png)
  - Prediction
  
  ![github-small](img/cfpred.png)
- Matrix Factorization
- Neural Network

### Files in the folder

- `data\`
  - `u.data`: raw file that contains all user ID, item ID and each corresponding rating
- `src\`
  - `cf\`: implementation of collaborative filtering model
  - `mf\`: implementation of matrix factorization model
  - `nn\`: implementation of neural network model
  
### Required packages
The code has been tested running under Python 3.7.3, with the following packages installed (along with their dependencies):
- tensorflow == 1.14.0
- numpy == 1.16.4
- sklearn == 0.21.2

### Running this code

Run collaborative filtering
```
$ cd src/cf
$ python main.py
```

Run matrix factorization
```
$ cd src/mf
$ python main.py
```

Run neural network
```
$ cd src/nn
$ python main.py
```
