# M6 Project - Video Surveillance for Road Traffic Monitoring

The goal of the project is to learn the basic concepts and techniques related to video sequences, which can be used for example for surveillance applications. The project is divided into 5 weeks and will include tasks such as background and foreground estimation, video stabilization and region tracking.

### **TEAM 1**

| Members     | Github |
| :---      | ---:       |
| Santiago Barbarisi |[SantiagoBarbarisi](https://github.com/SantiagoBarbarisi)|
| Lorenzo Betto |[BourbonCreams](https://github.com/BourbonCreams)|
| Raül Duaigües |[raulduaigues](https://github.com/raulduaigues)|
| Yevgeniy Kadranov|[YevKad](https://github.com/YevKad)|

## Week 1

### Task 1: Segmentation metrics. Understand precision & recall

- [X] Given the sequences TEST A and TEST B using different background substraction methods, implement and compute the evaluation measures 
    - True Positive
    - True Negative
    - False Positive
    - False Negative
    - Precision
    - Recall
    - F1 Score
    
- [ ] Explain the numerical results obtained and provide an interpretation of the different values

### Task 2: Segmentation metrics. Temporal analysis
- [X] Temporal analysis of the results. Create 2 graphs:

    - True Positive & Total Foreground pixels vs frame 
    - F1 Score vs frame
    
<!-- <p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2018-team1/blob/master/week1/TotalFG.png" width="500"/>
</p> -->

<!--  <p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2018-team1/blob/master/week1/F1_2.png" width="500"/>
</p> -->

- [ ] Explain and show why
        
### Task 3: Optical flow evaluation metrics
- [X] Optical flow estimations using the Lucas-Kanade algorithm (Sequences 45 and 157)
    - Metric: Mean Square Error in Non-occluded areas
    - Metric: Percentage of Erroneous Pixels in Non-occluded areas
    
- [ ] Discuss the obtained results and generate visualizations that help understanding them

### Task 4: De-synchornized results

- [X] Forward de-synchronized results for background substraction (Highway sequence)

<!-- <p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2018-team1/blob/master/week1/Des_TestA.png" width="500"/>
</p> -->

<!-- <p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2018-team1/blob/master/week1/Des_TestB.png" width="500"/>
</p> -->

### Task 5: Visual representation optical flow

- [X] Plot the optical flow
- [ ] Propose a simplification method for a clean visualization



