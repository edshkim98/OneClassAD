# OneClassAD
One-class learning project for anomaly detection using real industrial dataset <br>

1. The model is based on original CS-Flow model that has been modified. <br>
2. We tested our model against CS-Flow and Fastflow (current SOTAs) on our own benchmark datasets that are much larger than MVtech-AD (Camera lens, TCP boards)
3. We were able to increase the **inference speed for more than >2x** and outperforming CS-Flow by more than **>0.5% in AUROC** and **significant improvement in False Postitive Rate**.

**Dataset**
| |Camera Lens|TCP board|
| --- | --- | --- | 
|Train| 422 | 1432 |  
| Test| 802 | 2897 |


**Results**
| |Camera Lens (AUROC/FPR)|TCP board (AUROC/FPR)| Inf. speed (ms) |
| --- | --- | --- |--- |
|**Proposed**| **99.5/7.5%** | **99.9/26.9%** |  **36.5** |
|CS-Flow| 98.7/55.3% | 99.7/68.0% |  92.8 |
