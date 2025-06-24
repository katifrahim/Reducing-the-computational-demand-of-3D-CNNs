# Reducing the computational demand of 3D CNNs
A comparative analysis of Filter Pruning and Weight Pruning for 3D Convolutional Neural Networks (CNNs) 

## Research Question:
*To what extent can filter pruning effectively reduce the computational demand of 3D CNNs without sacrificing their performance for 3D object classification tasks, compared to weight pruning?*

## Introduction:
The use of 3D CNNs has been limited as their high computational demand hinders their deployment in resource-limited devices. Pruning is a compression-technique that aims to solve this by removing redundant parts of the model. Using pruning on 2D CNNs has been researched in depth, but less research has been done on using pruning on 3D CNNs due to their complexity and limited data availability in the past. 

I aim to fill this gap by comparing the impact of two popular pruning techniques (filter and weight pruning) on the computational demand and performance of a 3D CNN, specifically for 3D object classification tasks.

## Experiment Methodology
Primary experimental data is the main source of data in this paper. A 3D CNN was programmed and fed 3D object images from public datasets to perform 3D object classification. Afterwards, two copies of that model were created to perform the two pruning techniques on them. The computational demand (measured by GPU utilization) and the performance (measured by model accuracy) of these two models were then recorded at different compression levels. 
Due to the lack of relevant secondary data available to address the research question of this paper, I opted for an experimental methodology as it also offers flexibility of tailoring the variables to meet the exact needs of this paper.  
### The Dataset Used
ModelNet10 is a public dataset by Princeton university, which contains 4,889 labeled CAD models of 3D grayscale objects from 10 different categories, separated into 3,991 (80%) for training and 908 (20%) for testing. The CAD models present in this dataset are in the Object File Format (OFF). The object categories of this dataset are the most common object categories in the world, which includes bathtub, bed, chair, desk, dresser, monitor, night stand, sofa, table and toilet. 
### Preprocessing the dataset for use
I utilized python and a few of its libraries to convert the 3D objects present in this dataset to tensors. First, the Open3D library was used to convert the 3D objects to 3D mesh objects and then to voxel grids. Then the same library was used with the NumPy library to convert these voxel grids to arrays that contain coordinates of the occupied voxels of their respective voxel grids. These arrays were then converted to 3D arrays that contain binary values of the voxels of their respective voxel grids, using the same libraries. Next, the Pytorch library was used with these arrays to create custom Pytorch training and testing datasets with a tensor transformation to transform the arrays to tensors. The train-test-split size used was the official one suggested by the authors of the ModelNet10 dataset. Subsequently, these datasets were used to create their respective dataloaders with a batch size of 64. 
### The Dependent Variables
#### Computational Demand
The percentage of GPU utilized by the models during a single inference was taken as a metric to measure the computational demand of those models, as it provides a direct and quantitative measure of the computational demand of models. This was done by calculating the average batch GPU utilization during the testing phase of each model and then dividing it by the batch size to get the average GPU utilization per inference of the model. Since this metric depends on the GPU being used, I used the same GPU throughout the entire experimentation process to make sure that it accurately reflects the computational demand of the models.
#### Accuracy
The accuracies of the models were taken as a metric to measure their performance. The accuracy of each model was determined by calculating the percentage of accurate predictions made by the model for all classes of the testing dataset.
### The Convolutional Neural Network Programmed
The architecture of the programmed 3D CNN (see table 1) was finalized after several rounds of trial and error. Since the pruning models are copies of the original model, they share the same architecture.
![image](https://github.com/user-attachments/assets/0f4217f6-d1b2-44ca-87fe-bbf7e0c7838b)
### The Experimental Procedure
First, I created the original 3D CNN that is described in table 1 and then trained and tested it to calculate its accuracy and GPU utilization, which will be used to compare it with the other models. Following that, I created two copies of this original trained model and applied the two different pruning techniques 10 different times on them. Each time with an additional pruning percentage of 10%. In each iteration, the respective pruned model is retested to calculate its accuracy and GPU utilization at each pruning level. Moreover, the pruned models are not retrained for fine-tuning because it will cause the pruned model to regain the weights that were previously pruned by their respective pruning technique. 
In addition, it should be noted that the pruning percentage (or pruning level or compression level) of a model only reflects the percentage of weights and filters pruned in the convolution layers of that model.
## Experiment Results
The data present in the results contains large decimal places to give accurate results, as the differences between some values are really small to be noticed without large decimal places. 
### Tabular representation of data
A tabular representation of data was chosen because the results of the three models can be put in their three respective tables. 
#### Original model data
![image](https://github.com/user-attachments/assets/df3c0a55-e9c3-4922-8956-a8d7458a7b6f)
#### Weight pruning model data
![image](https://github.com/user-attachments/assets/6b9fad54-8199-40d4-a562-54ffc5c480a6)
#### Filter pruning model data
![image](https://github.com/user-attachments/assets/4c4753d3-3e18-4cea-9f86-fbc6b3fa7f27)
### Graphical representation of data
Graphical representation of data was chosen because it allows us to easily visualize and compare the accuracy and GPU utilization of both pruning models.
#### Accuracy
![image](https://github.com/user-attachments/assets/c2e940dc-82eb-48d3-9dbd-11e6965f0d27)
#### GPU Utilization
![image](https://github.com/user-attachments/assets/0c83fed9-be3e-4a1f-a72a-6e2643c09969)
### Result analysis
#### Analyzing accuracy
The results (see Graph 1) portray that there are many redundant weights present in the original 3D CNN model because there is not a significant change in the accuracy of both pruning models until 60% of their convolution layers’ weights are pruned.
According to the results, after pruning 60%, we can see noticeable fluctuations in the accuracy of both pruning models. The filter pruning continues to prune more redundant weights, due to which its accuracy increases until it has pruned 80% of the weights. Whereas, after pruning 60%, the weight pruning starts to prune the significant weights of the model, which causes its accuracy to fall. Similarly, after pruning 80%, the filter pruning also starts to prune the significant weights of the model, as a result causing its accuracy to fall as well.
Hence, filter pruning can prune up to 80% of the 3D CNN’s convolution layers’ weights without sacrificing its original accuracy. By contrast, weight pruning can prune up to 60% of the 3D CNN’s convolution layers’ weights while sacrificing just a little bit of its original accuracy.
#### Analyzing GPU utilization
Although the results (see Graph 2) showcase an overall reduction in the GPU utilization of the 3D CNN after pruning, there is still an irregular pattern in the GPU utilization of both pruning models. This is due to the irregular data access patterns that are caused by introducing sparsity into the models.
According to the results, for each pruning model, the GPU utilization is the lowest at its optimal pruning level where there is a turning point in its accuracy. The filter pruning can prune up to 80% with no marginal loss of accuracy and its lowest GPU utilization is also at 80% pruning. Therefore, the optimal pruning level of filter pruning is 80%. Similarly, the weight pruning can prune up to 60% with very less marginal loss of accuracy and its lowest GPU utilization is also at 60% pruning. For that reason, the optimal pruning level of weight pruning is 60%. The possible reason behind this is that, at these optimal pruning levels, the respective pruning technique prunes most of the redundant weights of the convolution layers while retaining most of the significant weights of the convolution layers. This causes the accuracy of the pruning models to be maximized and its computational demand to be minimized at these specific optimal pruning levels.
This illustrates that, even though the GPU utilization itself does not have a clear pattern, there is still a clear pattern present between the GPU utilization and the accuracy of the pruning techniques at their respective optimal pruning levels.
Additionally, it should be noted that the value of the GPU utilization results significantly depends on the type of GPU being utilized during the inference. Since I used a powerful GPU during the experimentation, known as the Nvidia Tesla T4 GPU (offered by Google Colab), the GPU utilization values I obtained are really small. Thus, it accurately reflects the computational demand of the models, which is the purpose of calculating it in the first place.
### Evaluation
The results show a clear trade-off between both pruning techniques in terms of their accuracy, GPU utilization and optimal pruning level. The filter pruning technique is more accurate than weight pruning. This is because, according to the results, filter pruning can prune 80% of the 3D CNN’s convolution layers’ weights without sacrificing any accuracy, while weight pruning can only prune 60% of the 3D CNN’s convolution layers’ weights even with sacrificing a little bit of accuracy. On the opposite side, the GPU utilization of the weight pruning is less than that of the filter pruning, which means that the model obtained from weight pruning is less computationally demanding than that obtained from filter pruning. This is because, according to the results, the GPU utilization of the weight pruning at its optimal pruning level (60%) is almost two times lower than that of the filter pruning at its optimal pruning level (80%).
## Limitations and Future Research Opportunities
### Using a different pruning metric
In this paper, I utilized the L1 norm metric for both pruning techniques, as it is the most commonly used one for CNNs. However, there are many other metrics to calculate the same pruning techniques as well. Therefore, it would be interesting to compare the effect of using weight and filter pruning with a different metric for 3D CNNs.
### Using multiple performance and computational demand metrics
In the experimentation process of this research paper, I used single metrics to determine the performance and computational demand of the 3D CNN models. However, relying on single metrics to measure each dependent variable of the 3D CNN models might not give completely accurate results and the findings of the paper significantly depends on the dependent variables. Thus, using multiple metrics to measure the dependent variables of this research paper can give more accurate results, allowing the findings of this paper to further improve.
### Using a different dataset
Although the dataset I used in the experimentation process of this paper contains 3D images, it is still not very complex because the 3D images are in grayscale color, which means they have one channel . Hence, It would be interesting to use a colored 3D image dataset for the research done in this paper, as it will have 3 channels. This will make both the dataset and the 3D CNN model much more complex, significantly increasing its computational demand.
### Reducing the pruning gap
In my experiment, I pruned 0% to 100% for each pruning technique with a gap of 10%. This is a really big gap, which might cause the obtained optimal pruning level for each pruning technique to not be totally accurate. Thus, using a smaller pruning gap can allow the findings of this paper to further improve by increasing its accuracy.
## Conclusion
In this paper, the effect of applying filter pruning and weight pruning on 3D CNNs to enhance their computational efficiency, without sacrificing any performance, was compared and analyzed with logical and mathematical explanations, specifically to perform 3D object classification tasks.
The results show that there is a clear trade-off present between both pruning techniques in terms of their performance, computational demand and optimal pruning level. The filter pruning, at its optimal pruning level, can prune more weights from the convolution layers of a 3D CNN with no marginal loss of accuracy, but it has higher computational demand. By contrast, the weight pruning, at its optimal pruning level, can prune less weights from the convolution layers of a 3D CNN while sacrificing some accuracy, but it has lower computational demand. Hence, directly stating one pruning technique superior or inferior to the other one will not be a good approach to address the research question of this paper, as there are obvious trade-offs present between both techniques in terms of their performance, computational demand and optimal pruning level.
Hopefully, this paper will prove useful to CNN researchers and developers in reducing the computational demand of 3D CNNs and allowing them to be deployed in resource-limited devices. Hence, increasing its accessibility and resulting in advancements in the countless fields that utilize it.
