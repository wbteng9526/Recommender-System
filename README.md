## Recommender System

This repository is created for comparing the 3 models for building up a recommender system:
- Collaborative filtering
- Matrix factorization
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

### Running this code
