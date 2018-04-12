# M6 Project - Video Surveillance for Road Traffic Monitoring

The goal of the project is to learn the basic concepts and techniques related to video sequences, which can be used for example for surveillance applications. The project is divided into 5 weeks and will include tasks such as background and foreground estimation, video stabilization and region tracking.

### **TEAM 1**

| Members     | Github | Mail |
| :---      | ---:       | ---: |
| [Santiago Barbarisi](https://www.linkedin.com/in/santiago-barbarisi-abb79787/) |[SantiagoBarbarisi](https://github.com/SantiagoBarbarisi)| santiago.barbarisi@e-campus.uab.cat |
| [Lorenzo Betto](https://www.linkedin.com/in/lorenzo-betto/) |[BourbonCreams](https://github.com/BourbonCreams)|   lorenzo.betto@e-campus.uab.cat  |
| [Raül Duaigües](https://www.linkedin.com/in/ra%C3%BCl-duaig%C3%BCes-84943b103/) |[raulduaigues](https://github.com/raulduaigues)|    raul.duaigues@e-campus.uab.cat  |
| [Yevgeniy Kadranov](https://www.linkedin.com/in/yevkad/)|[YevKad](https://github.com/YevKad)|   yevgeniy.kadranov@e-campus.uab.cat   |



### Code execution

Every week's submission will be placed in a different folder called 'weekX', where X is the number of the corresponding week. Each of these folders contains a 'main.py' file that can be ran to execute the submitted code.


## Week 1: Introduction, DB and Evaluation metrics

### Task 1: Segmentation metrics. Understand precision & recall

- [X] Given the sequences TEST A and TEST B using different background substraction methods, implement and compute the evaluation measures 
    - True Positive
    - True Negative
    - False Positive
    - False Negative
    - Precision
    - Recall
    - F1 Score
    
- [X] Explain the numerical results obtained and provide an interpretation of the different values

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

- [X] Explain and show why
        
### Task 3: Optical flow evaluation metrics
- [X] Optical flow estimations using the Lucas-Kanade algorithm (Sequences 45 and 157)
    - Metric: Mean Square Error in Non-occluded areas
    - Metric: Percentage of Erroneous Pixels in Non-occluded areas
    
- [X] Discuss the obtained results and generate visualizations that help understanding them

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
- [X] Propose a simplification method for a clean visualization

## Week 2:


### Task 1: Gaussian modelling

- [X] Gaussian function to model each background pixel 
    - First 50% of the test sequence to model background
    - Second 50% to segment the foreground
    
- [X] Evaluate using Precision, Recall, F1-score vs alpha
- [X] Evaluate using Precision vs Recall curve and Area Under the Curve (AUC)

### Task 2: Adaptive modelling

- [X] Gaussian function to model each background pixel 
    - First 50% frames for training
    - Second 50% left background adapts
    
- [X] Best pair of values (𝛼, ⍴) to maximize F1-score
    - Obtain first the best 𝛼 for non-recursive, and later estimate ⍴ for the recursive cases
    
- [X] Compare both the adaptive and non-adaptive version and evaluate them for all 3 sequences proposed using F1 score/AUC


### Task 3: Comparison with state-of-the-art

- [X] Evaluate precision vs recall
- [X] Evaluate the sequences than benefit more one algorithm and explain why

### Task 4: Color Sequences
- [X] Update your implementation to support color sequences


## Week 3:


### Task 1: Hole filling

- [X] Post-process Week2 with hole filling
    - Report with the AUC & gain for each of the three video sequences

### Task 2: Area Filtering

- [X] Plot a graph of AUC vs pixels for each sequence and discuss some qualitative examples.

- [X] Post-process the best configuration from Task 1, find the P with highest mean AUC for each sequence and discuss qualitative results

### Task 3: Additional morphological processings

- [X] Explore with other morphological filters and combinations to improve AUC for foreground pixels.

    - Closing
    - Opening
    - Dilation
    - Erosion
    - Different structural elements...

### Task 4: Shadow removal
- [X] Search for existing techniques, run/implement at least one and assess its performance

### Task 5: Evaluation
- [X] Summarize which is your best configuration and show it improves the performance from previous week:
    - Compare the precision / recall curves
    - Update the AUC & compute the gain.



## Week 4:


### Task 1: Optical Flow

- [X] Implement a Block Matching solution for optical flow estimation
    - Forward or Backward compensation
    - Area of Search
    - Size of the blocks

- [X] Compare Block Matching vs. Other techniques

### Task 2: Video Stabilization

- [X] Video Stabilization with Block Matching
    - Use the estimated flow between two frames to align them. Apply it on the 'Traffic' sequence
    
- [X] Compare Video Stabilization with Block Matching vs. Other techniques
    - Compute PR curve and AUC to compare
    
- [X] Stabilize your own video
    




## Week 5:


### Task 1: Vehicle tracker

- [X] Implement a vehicle tracker with Kalman filter

- [X] Implement a vehicle tracker with other tools (KCF and MEDIANFLOW from OpenCV were tested)

### Task 2: Speed estimator

- [X] Estimate the speed of the vehicles.





## External Links:

- <a href="https://drive.google.com/open?id=10gGRAkTmED8FXuEFtYekMw7f72QMubzpMk8xuwhyKJQ">Final presentation (Google Drive) </a>
- <a href="https://www.overleaf.com/read/xgwgswcmsqqg">Final report (OverLeaf) </a>
