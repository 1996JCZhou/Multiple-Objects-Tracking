# Kalman-Filter in Multi-object Tracking, along with two Matching algorithms: the Hungarian algorithm and the Max Weight Matching algorithm

Welcome to my GitHub project that delves into the fascinating realm of Multi-object Tracking (MOT) for videos, employing the powerful Kalman Filter alongside two efficient Matching algorithms: the Hungarian algorithm and the Max Weight Matching algorithm. Multi-object Tracking (MOT) is a crucial task in computer vision and robotics, aimed at accurately estimating the trajectories of multiple objects over time from noisy and uncertain measurements. The Kalman Filter, a widely-used recursive algorithm, plays a pivotal role in state estimation, offering an optimal solution for tracking dynamic systems under uncertainty.

The kinematic system is characterized by a linear, discrete-time, vectorial, and time-variant model. Within this framework, we assume that all target objects in the video move in straight lines with uniform velocity during dynamic modeling. 

The primary objective of this project is to utilize the Kalman Filter's capabilities, a recursive estimation algorithm, to track multiple objects in video sequences accurately. The Kalman Filter works by predicting the state of the system using kinematic modeling and then updating the state based on observed measurements, which, in this case, are bounding box positions obtained from a reliable detector such as the YOLO-Family.

The challenge lies in effectively matching these predicted and detected bounding box positions within each video frame. This is where the Hungarian algorithm and the Max Weight Matching algorithm come into play. By leveraging these matching algorithms and calculating the Intersection over Union between bounding boxes, we obtain matched pairs of predicted and detected bounding box positions. With the matched detected bounding box positions, we utilize them as observation values for the Kalman Filter's filtering process to update the system's state, enhancing the accuracy of our Multi-object Tracking. 

I have consciously chosen not to directly link detected bounding box positions frame-to-frame to form trace tracks due to two key reasons. Firstly, target objects can be occluded, leading to missing detected bounding boxes and incomplete trace tracks. Secondly, even reliable detectors produce noisy observations, necessitating the use of the Kalman Filter's kinematic modeling to account for uncertainties and refine the tracking process.

In this GitHub repository, you will find a comprehensive collection of code, implementation examples, and resources that will enable you to explore and apply these techniques to your own Multi-object Tracking projects.

## Requirements
- python
- networkx
- opencv-python
- scipy

## Results



