This folder contains the Standard PRM and ANC PRM implementations. 
The results of these motion planning is printed and visualized. 

Note: All these implementations are done in serial. The parallel implementations can be found in their respective Google Colab notebooks under in the "Benchmarking" folder. 

This implementation is specific to a simplistic 2-D plane configuration space.
The structure and skeleton code of this work was derived from Professor Plancher's 
Fall 2022 - Robotics Course - RRT Homework. 

There are four test cases with different configuration spaces. 
They are captured in the four "run____.py" files. 
Each can be run in both with Standard PRM and ANC PRM. 
To change with algorithm is used, simply comment/uncomment out the relevant line the "run___.py" file. 

The returned output is just normal printed statements to standard output. 
The visualized result is displayed in a seperate window with PyGame.
