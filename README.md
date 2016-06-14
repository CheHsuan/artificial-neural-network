# artificial-neural-network
It's a framework of 3-hidden layer artificial neural network. Users can define their own network and input set easily.


# Usage
## How to compile
make 
## Excute
./model xxx.xml xxx.txt xxx.txt

The first first file is an xml file which defines the nerual network architecture containing the neurons number in each layers, the learming rate, epoch time, activation function, etc.

The second argument indicates the path of training set and the third argument is the path of validation set. The format in data set should be expressed like the following example.

*desired_output*&nbsp;&nbsp;&nbsp;&nbsp;*attribute_1*&nbsp;&nbsp;&nbsp;&nbsp;*attribute_2*&nbsp;&nbsp;&nbsp;&nbsp;*attribute_3*&nbsp;&nbsp;&nbsp;&nbsp;*...*&nbsp;&nbsp;&nbsp;&nbsp;*attribute_N*  
1&nbsp;&nbsp;&nbsp;&nbsp;0.1&nbsp;&nbsp;&nbsp;&nbsp;0.25&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;1.89  
0&nbsp;&nbsp;&nbsp;&nbsp;0.4&nbsp;&nbsp;&nbsp;&nbsp;-2.25&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;1.59  
0&nbsp;&nbsp;&nbsp;&nbsp;0.2&nbsp;&nbsp;&nbsp;&nbsp;0.15&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;1.31  
1&nbsp;&nbsp;&nbsp;&nbsp;2.1&nbsp;&nbsp;&nbsp;&nbsp;1.70&nbsp;&nbsp;&nbsp;&nbsp;-1&nbsp;&nbsp;&nbsp;&nbsp;1.77

The desired output is the classification of entity. Only the **integer** is accepted and should start from zero.
The attributes can be an **integer**, a **float** or a **double**. 
# Example 
./model netdefine.xml ./dataset/flower.txt ./dataset/flower.txt

We use the same data set as training set and validation set in this case. The data set is about iris downloaded from [here](http://archive.ics.uci.edu/ml/datasets/Iris). The program will print out the accuracy and the loss during training and you can observe the improvement easily. Now, it is your turn to train your own model with different data set whatever you like. Go and give it a try.
