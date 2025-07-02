# Reducing the computational demand of 3D CNNs
A comparative analysis of Filter Pruning and Weight Pruning for 3D Convolutional Neural Networks (CNNs) 

## Research Question
*To what extent can filter pruning effectively reduce the computational demand of 3D CNNs without sacrificing their performance for 3D object classification tasks, compared to weight pruning?*

## Introduction
The high computational demand of 3D CNNs limits their deployment on low-resource devices like Single Board Computers (SBCs). Pruning is a technique that solves this by removing redundant model-parts. While pruning on 2D CNNs has been well-researched, its application on 3D CNNs remains underexplored due to their complexity and data limitations. 

This project compares filter and weight pruning to evaluate their effect on the the computational demand (GPU utilization) and performance (accuracy) of 3D CNNs for 3D object classification tasks.

The entire data pre-processing and experiment was done using **Python** in **Google Colab** with the **Nvidia T4 GPU**.
## Dataset Used
[ModelNet10](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset) is a public dataset by Princeton university, which contains 4,889 labeled CAD models of 3D grayscale objects from 10 different categories, separated into 3,991 (80%) for training and 908 (20%) for testing. 

![image](https://github.com/user-attachments/assets/97b2954e-109f-4598-b716-bb110e8f69dd)
## Data Pre-Processing
Converting the 3D-CAD models to tensor dataloaders: (Related: [Voxelization](https://github.com/katifrahim/Voxelization))
- **Open3D library:** 3D-CAD models -> 3D mesh objects -> Voxel grids  ![plane](https://github.com/user-attachments/assets/6cad2ccf-02b2-461e-ba08-d77087308448)
- **NumPy library:** Voxel grids -> Arrays containing coordinates of the occupied voxels -> 3D arrays containing binary values representing voxels
- **Pytorch library:** 3D arrays containing binary values representing voxels -> Training and testing datasets with a tensor transformation -> Dataloaders with a batch size of 64. 
## Dependent Variables
- **Computational demand (GPU utilization):** Measured by calculating the average batch GPU utilization during the testing phase of each model and then dividing it by the batch size to get the average GPU utilization per inference of the model.
- **Performance (Accuracy):** Measured by calculating the percentage of accurate predictions made by the model for all classes of the testing dataset.
## Experiment
- Used Pytorch to create a parent 3D CNN with the following architecture: ![image](https://github.com/user-attachments/assets/083ced96-2536-44a8-abf5-1ab5bf291088)
- Trained and tested this model to calculate its accuracy and GPU utilization.
- Created two child copies of this trained parent model and then applied the two different pruning techniques 10 different times on them (each time with an additional pruning percentage of 10%).
- For each pruning level, the GPU utilization and accuracy of the respective pruned model was recorded to be compared. 
- **Note:** The pruning percentage (or pruning level) of a model reflects the percentage of weights and filters pruned in the convolution layers of that model.
## Results

### Tabular data representation
#### Original model's data
![image](https://github.com/user-attachments/assets/63b25138-67a5-4e7e-91a2-59741790f8c0)
#### Weight pruning model's data
![image](https://github.com/user-attachments/assets/ed902a6c-1617-4eb1-a6b5-c3ab7baa034a)
#### Filter pruning model's data
![image](https://github.com/user-attachments/assets/c7261d31-fbff-4283-a5e5-6ec1aa3e2fbd)
### Graphical representation of data
#### Accuracy
![image](https://github.com/user-attachments/assets/d1b6647c-9367-4763-9e6f-a6c1c73ecfa9)
#### GPU Utilization
![image](https://github.com/user-attachments/assets/39080739-52a0-45de-88d6-8a9f83c62cfc)
## Evaluation
The results show a clear trade-off between both pruning techniques in terms of their accuracy, GPU utilization and optimal pruning level:
- **Filter pruning leads to more accuracy than weight pruning:** Filter pruning can prune 80% of the 3D CNN’s convolution layers’ weights at max without sacrificing accuracy, while weight pruning can only prune 60% of these weights at max even with small accuracy-loss.
- **Weight pruning leads to lesser GPU utilization than filter pruning:** The GPU utilization of the weight pruning at its optimal pruning level (60%) is almost two times lower than that of the filter pruning at its optimal pruning level (80%).
## Future Research Opportunities
- Using a different pruning metric (I used the L1 norm metric, but other metrics can be used too).
- Using multiple performance and computational demand metrics.
- Using a different dataset.
- Reducing the pruning gap to less than 10% to find more accurate optimal pruning levels.
## Conclusion
The results show a clear trade-off between both pruning techniques in terms of their performance, computational demand and optimal pruning level. The filter pruning, at its optimal pruning level, can prune more weights from the convolution layers of a 3D CNN with no marginal loss of accuracy, but it has higher computational demand. By contrast, the weight pruning, at its optimal pruning level, can prune less weights from the convolution layers of a 3D CNN while sacrificing some accuracy, but it has lower computational demand. Hence, directly stating one pruning technique superior or inferior to the other one will not be a good approach to address the research question.
