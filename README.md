# SARSS
To ensure that readers and reviewers can verify our results, our system is built directly upon the widely adopted FAST-LIO2 framework. 
It shares the exact same runtime environment and dependencies (e.g., ROS, PCL, Eigen). 
Consequently, users who have successfully run FAST-LIO2 require no additional library installations. 
The package can be seamlessly integrated into a standard ROS workspace: 
users simply need to download the code and compile it using the standard catkin_make command, following the identical procedure as FAST-LIO2.

We have provided pre-configured launch files and sample datasets (rosbags) in the repository. 
Once compiled, users can directly reproduce the experiments by executing the launch file and playing the provided dataset via rosbag play *.bag.

usage:rosbag play *.bag

You can download the dataset through this link:

https://drive.google.com/file/d/1Bm_d6N9bqcMlkTfR-9MvIww6r1zF4TYL/view?usp=sharing

1.roslaunch fast_lio mapping_velodyne.launch
2.rosbag play *.bag
or other dataset
2.rosbag play hku_campus_seq_00.bag -s 115 -u 40
