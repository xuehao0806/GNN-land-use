# GNN-land-use

## Update on 2024-03-20
## Summary
- **Adding Hetero-edges information**
- **MinMax Normalisation**

**RGCN**

| RGCN       | MSE   | RMSE  | MAE   | R2    |
|------------|-------|-------|-------|-------|
| office     | 0.003 | 0.051 | 0.035 | 0.915 |
| sustenance | 0.002 | 0.044 | 0.031 | 0.939 |
| transport  | 0.003 | 0.054 | 0.039 | 0.927 |
| retail     | 0.004 | 0.065 | 0.046 | 0.865 |
| leisure    | 0.005 | 0.069 | 0.051 | 0.844 |
| residence  | 0.005 | 0.070 | 0.051 | 0.866 |

**NN**

|  NN        | MSE   | RMSE  | MAE   | R2    |
|------------|-------|-------|-------|-------|
| office     | 0.015 | 0.124 | 0.086 | 0.484 |
| sustenance | 0.013 | 0.116 | 0.079 | 0.581 |
| transport  | 0.019 | 0.139 | 0.097 | 0.514 |
| retail     | 0.018 | 0.134 | 0.103 | 0.417 |
| leisure    | 0.020 | 0.141 | 0.106 | 0.342 |
| residence  | 0.025 | 0.158 | 0.123 | 0.310 |

**GCN**

| GCN        | MSE   | RMSE  | MAE   | R2    |
|------------|-------|-------|-------|-------|
| office     | 0.004 | 0.060 | 0.043 | 0.881 |
| sustenance | 0.002 | 0.046 | 0.033 | 0.935 |
| transport  | 0.004 | 0.063 | 0.046 | 0.900 |
| retail     | 0.006 | 0.077 | 0.057 | 0.812 |
| leisure    | 0.007 | 0.086 | 0.064 | 0.753 |
| residence  | 0.007 | 0.083 | 0.062 | 0.809 |

**GAT**

|  GAT       | MSE   | RMSE  | MAE   | R2    |
|------------|-------|-------|-------|-------|
| office     | 0.003 | 0.054 | 0.039 | 0.904 |
| sustenance | 0.002 | 0.039 | 0.028 | 0.952 |
| transport  | 0.003 | 0.054 | 0.039 | 0.927 |
| retail     | 0.005 | 0.068 | 0.050 | 0.851 |
| leisure    | 0.006 | 0.080 | 0.058 | 0.789 |
| residence  | 0.006 | 0.074 | 0.054 | 0.847 |

**GraphSage**

|  GraphSage | MSE   | RMSE  | MAE   | R2    |
|------------|-------|-------|-------|-------|
| office     | 0.003 | 0.058 | 0.043 | 0.887 |
| sustenance | 0.002 | 0.045 | 0.032 | 0.938 |
| transport  | 0.003 | 0.059 | 0.043 | 0.912 |
| retail     | 0.005 | 0.070 | 0.053 | 0.841 |
| leisure    | 0.007 | 0.083 | 0.060 | 0.773 |
| residence  | 0.006 | 0.078 | 0.058 | 0.831 |
## Update on 2024-03-18
- **Name**: Xuehao
- **Date of Update**: 2024-03-18

## Summary
- **Change the way for collection of 'Residence'**
- **Update the road network graph**
- **Normalised each label in a same way**

## Performance

- NN
 - No edge-related information.

| Category    | MSE   | RMSE  | MAE   | R2    |
|-------------|-------|-------|-------|-------|
| office      | 0.503 | 0.709 | 0.460 | 0.475 |
| sustenance  | 0.418 | 0.647 | 0.424 | 0.577 |
| transport   | 0.499 | 0.706 | 0.477 | 0.487 |
| retail      | 0.588 | 0.767 | 0.577 | 0.430 |
| leisure     | 0.658 | 0.811 | 0.603 | 0.309 |
| residence   | 0.676 | 0.822 | 0.640 | 0.324 |

 - GCN.
 - No edge-related information.

|            | MSE   | RMSE  | MAE   | R2    |
|------------|-------|-------|-------|-------|
| office     | 0.137 | 0.371 | 0.265 | 0.856 |
| sustenance | 0.097 | 0.312 | 0.219 | 0.901 |
| transport  | 0.148 | 0.385 | 0.274 | 0.847 |
| retail     | 0.194 | 0.440 | 0.325 | 0.811 |
| leisure    | 0.254 | 0.504 | 0.374 | 0.735 |
| residence  | 0.205 | 0.453 | 0.334 | 0.795 |

 - GraphSage.
 - No edge-related information.

| Category    | MSE   | RMSE  | MAE   | R2    |
|-------------|-------|-------|-------|-------|
| office      | 0.109 | 0.330 | 0.240 | 0.886 |
| sustenance  | 0.068 | 0.261 | 0.185 | 0.931 |
| transport   | 0.112 | 0.335 | 0.240 | 0.884 |
| retail      | 0.170 | 0.412 | 0.309 | 0.835 |
| leisure     | 0.230 | 0.480 | 0.361 | 0.759 |
| residence   | 0.187 | 0.433 | 0.322 | 0.813 |

## Summary

- **Adding simple forward neural network as baseline model**

- **Using the previous strategy of processing data splitting**

## Performance
 - Simple Neural Network.
 - With same parameter setting of other models.
 - No edge-related information.

| Category    | MSE   | RMSE  | MAE   | R2    |
|-------------|-------|-------|-------|-------|
| office      | 0.156 | 0.396 | 0.270 | 0.423 |
| leisure     | 0.089 | 0.298 | 0.223 | 0.366 |
| transport   | 0.173 | 0.417 | 0.288 | 0.452 |
| retail      | 0.219 | 0.469 | 0.353 | 0.435 |
| sustenance  | 0.321 | 0.567 | 0.373 | 0.552 |
| residence   | 0.115 | 0.340 | 0.223 | 0.175 |


 - GraghSage (with Neighbor-sampling) 
 - No isolation on test nodes

| Category    | MSE   | RMSE  | MAE   | R2    |
|-------------|-------|-------|-------|-------|
| office      | 0.040 | 0.199 | 0.143 | 0.852 |
| leisure     | 0.041 | 0.202 | 0.151 | 0.706 |
| transport   | 0.048 | 0.218 | 0.161 | 0.848 |
| retail      | 0.071 | 0.266 | 0.202 | 0.817 |
| sustenance  | 0.058 | 0.242 | 0.170 | 0.918 |
| residence   | 0.050 | 0.225 | 0.141 | 0.639 |

## Update on 2024-03-02
- **Name**: Xuehao
- **Date of Update**: 2024-03-02

## Summary

- **Mini-batch Training Process**: Implemented a mini-batch method for graph data.

- **Multiple Sampling Methods**: The update provides options for different sampling processes, allowing the selection between 'RandomNodes' and 'NeighborNodes'. 

- **Isolation of Test Data**: To prevent data leakag. the test data has been isolated from the training dataset, following the suggestion from Adam.


## Performance

 - GraghSage (with Neighbor-sampling) 
 - Batch size of 64 and epoch of 200.

|            | MSE   | RMSE  | MAE   | R2    |
|------------|-------|-------|-------|-------|
| office     | 0.078 | 0.278 | 0.175 | 0.714 |
| leisure    | 0.053 | 0.231 | 0.167 | 0.620 |
| transport  | 0.067 | 0.259 | 0.176 | 0.788 |
| retail     | 0.107 | 0.327 | 0.230 | 0.725 |
| sustenance | 0.113 | 0.336 | 0.209 | 0.842 |
| residence  | 0.086 | 0.293 | 0.168 | 0.390 |

![training process](visualisation/loss_plot.png)