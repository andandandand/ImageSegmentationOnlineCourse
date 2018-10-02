# Course: Image Segmentation with Python

### Course developer and instructor: [Antonio Rueda-Toicen, MSc.](https://www.linkedin.com/in/antonioruedatoicen/)
+ Chief Technology Officer at [The Chain](http://thechain.tech/)
+ Technical expert and data science mentor at [Thinkful](https://www.thinkful.com/)
+ Research programmer at the [Algorithmic Dynamics Lab](https://www.algorithmicdynamics.net/)
+ Instructor at the National Institute of Bioengineering (INABIO) [Universidad Central de Venezuela](http://www.ucv.ve/)

emails (use only one): antonio.rueda.toicen@gmail.com, atoicen@thinkful.com, antonio.rueda@ucv.ve, antonio.rueda.toicen@algorithmicnaturelab.org, art@wek.io

**homepage: www.digital-spaceti.me**

### Python Technology stack

+ NumPy
+ TensorFlow and Keras
+ Sckit-image
+ OpenCV
+ Google's Colaboratory


### Related courses

+ [Deep Learning in Python](https://www.datacamp.com/courses/deep-learning-in-python)
+ [Convolutional Neural Networks for Image Processing](https://www.datacamp.com/courses/convolutional-neural-networks-for-image-processing)
+ [Biomedical Image Analysis with Python](https://www.datacamp.com/courses/biomedical-image-analysis-in-python)


## What problems will students learn to solve?

+ Perform instance segmentation by implementing the U-net deep neural network in TensorFlow 
+ Segment multimodal images in a reproducible manner with the seeded GrowCut algorithm (implemented on Numpy)
+ Assess the quality of a segmentation quantitatively using the f1, IOU and Rand error measures

## What techniques or concepts will students learn?

Students will learn the functioning of the state-of-the-art U-net deep neural network for image segmentation, including:
  * The network's encoder-decoder architecture
  * Activation functions, convolutions, and pooling
  * Upsampling and transposed convolutions
  * Use of the ADAM and SGD optimizers
  * Understanding the processing done in the final output layers 
  * Performing data augmentation the network more robust 
  * Its implementation in TensorFlow
  
Students will also learn the use of the "conventional" yet powerful GrowCut algorithm for seeded segmentation, including:
  * A workflow for selecting segmentation seeds comprising the use of entropy minima, and the k-medoids algorithm
  * Its implementation in both Numpy and TensorFlow
  
Finally students will be able to quantitatively assess the quality of a segmentation, computing the following metrics:
  * f1-measure (aka Dice similarity coefficient)
  * Rand error
  * warping error 
  
**Students will segment features from biomedical, satellite and conventional image datasets.**

## Who is this course for?

* "Advanced Alex", a programmer or data scientist with an understanding of image analysis and/or statistics who has probably taken one or more of the related courses in DataCamp: 

  + [Deep Learning in Python](https://www.datacamp.com/courses/deep-learning-in-python)
  + [Convolutional Neural Networks for Image Processing](https://www.datacamp.com/courses/convolutional-neural-networks-for-image-processing)
  + [Biomedical Image Analysis with Python](https://www.datacamp.com/courses/biomedical-image-analysis-in-python)

# Course Outline

## Part 1: Segmenting images and quantifying the quality of results

+ **Lesson 1.1 - Introduction to image segmentation**
  * **A learning objective: segment a tumor from healthy tissue using a ground truth mask**
  
    + Intro video explaining what image segmentation is (3 min)
  
     ![Ground truth image](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/TumorSim%20ground%20truth.png)

    + Drill: Loading an image with OpenCV
  
    + Drill: Importing medical image data with PyDICOM
        + Importing a TumorSim T2 MRI texture
        + Importing a labeled ground truth dataset
        
     + Video explaining what ground truth is and how to use it as a class mask (3 min)
    
    + Drill: Using a segmentation mask to select regions of an image
        + Select the tumor area in 2D brain MRI with an ndarray multiplication
        + Select the tumor volume in 3D brain MRI with an ndarray multiplication
        + Visualizing the tumor region in 3D
          
    ![3D GrowCut](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/3d-GrowCut-brain.PNG) 
    
+ Multiple choice quiz about image segmentation (purpose of segmentation, use of masks)
  
+ Video explaining segmentation with the GrowCut cellular automaton (3 min)
 
 + **Lesson 1.2 - Seeded segmentation** 
    * **A learning objective: segment an image using a Numpy implementation of the GrowCut cellular automaton** 
    
      + Identify and use entropy minima as segmentation seeds
      + Label entropy minima with k-medoids
      + Define minima area for segments to avoid oversegmentation
    ![K-medoids brain clustering](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/seeded%20segmentation.PNG)

+ **Lesson 1.3 - Evaluating a segmentation's quality quantitatively** 
  * **A learning objective: compute the main metrics used to assess the quality of image segmentation**

  + Video explaining the quantitative assessment of a segmentation's quality (3 min)

  + Evaluating the quality of a segmentation
    + Count true positives, true negatives, false positives, and false negatives 
    + Compute precision
    + Compute recall
    + Compute the f1-measure
    + Compute Intersection over Union (IoU)
    + Compute the warping error
    + Compute the Rand error 


## Part 2: Implementing U-net (part i)
**Use case shown: segmentation of neurons in electron microscopy **
![segmented neurons](http://brainiac2.mit.edu/isbi_challenge/sites/default/files/Challenge-ISBI-2012-sample-image.png)


+ **Lesson 2.1 - Introduction to deep learning for image segmentation**

 **A learning objective:  mapping the activation of layers in convolutional neural networks to features in order to perform instance segmentation.** 

+ Intro video explaining how convolutional networks can perform instance segmentation (3 min)


![texto alternativo](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

 + **Lesson 2.2 - The U-net network for instance segmentation**

**A learning objective: understand the encoder-decoder architecture of the U-net network and the use of the ReLU activation function**  

+ U-net architecture intro video (3 min)
  + Quick multiple choice quiz about U-net's architecture (purpose of U-net, contents of layers, what convolution is, what upsampling is)

+ Video explaining activation functions in U-net (3 min)  
    
![](https://cdn-images-1.medium.com/max/1600/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg)  

+ Drill: Implement Regular ReLU
+ Drill: Implement Leaky ReLU


+ **Lesson 2.3 - Computing image convolutions and pooling operations** 

**A learning objective: apply convolutions and pooling operations to images and intermediate layers of a U-net network**


+ Video explaining the use of convolutions and pooling (3 min)

![image convolution](https://i.stack.imgur.com/YDusp.png)
+ Applying image convolutions
  + Applying a convolution filter
  + Changing the weights of a convolution filter
  
+ Applying pooling
  * Implement max pooling

+ **Lesson 2.4 - Performing image augmentations on the training data** 

**A learning objective: perform data augmentation to increase the amount of labeled data fed to the network.**

+ Video explaining the need of image cropping and data augmentation in U-net (3 min)

+ Doing data augmentation
  + Mirroring
  + Rotation
  + Vertical/horizontal flips 
  + Deformations
  + Doing image cropping


## Part 3: Implementing U-net (part ii)
**Use case shown: segmentation of road images in RGB and RGBI images**

![](https://github.com/aschneuw/road-segmentation-unet/blob/master/report/figures/submission_selection/images_003.png?raw=true)

**Lesson 3.1: The decoder part of the U-net architecture**
+ **A learning objective: Implement the decoder part of the U-net architecture**

* Video explaining the decoder part of the U-net learning architecture, explains upsampling, mentions what unpooling and transposed convolutions are and their purpose (3 min)

**Lesson 3.2: Unpooling operations on U-net**
+ **A learning objective: apply unpooling operations on a U-net layer.** 

+ Applying unpooling 
  + Replacing each entry with an $𝑛 \times 𝑛$ matrix filled with the original entry (NumPy drill).
  + Replacing each entry with an $𝑛 \times 𝑛$ matrix with the original entry in the upper left and the other squares set to 0. [1506.02753]   (NumPy drill)
 
**Lesson 3.3 Computing transposed convolutions for upsampling**
![Transposed convolution](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/transposed-convolutions.PNG)  
 **A learning objective:  compute a transposed convolution with tf.nn.conv2d_transpose**
  
  + Video on transposed convolution (3 min)
  
  + Drill: use of tf.nn.conv2d_transpose

  

## Part 4: Implementing U-net (part iii)
**Use case shown: segmentation of multispectral satellite images**


![](https://camo.githubusercontent.com/28891f9cc505787a642b1ddabdc162e236c26204/68747470733a2f2f692e696d6775722e636f6d2f686a49546670632e706e67)

**Lesson 4.1: Output layers of U-net**

**A learning objective: Tweaking the output of the final layers of U-net.**

+ Video explaining the workings of the output layer of U-net (3 min)

+ Quiz on the output layer

**Lesson 4.2: Tradeoffs in loss functions**
**A learning objective: compute different loss functions and observe the effect of the choice on the resulting segmentation**

+ Video explaining loss functions (3 min)

+ Quiz on loss functions 

+ Computing loss functions:
  + Computing Cross entropy
  + Computing Mean squared error
  + Computing Mean absolute error
  + Using the Dice similarity coefficient as an error metric

**Lesson 4.3: Choosing an optimizer for U-net**

**A learning objective:  Evaluate tradeoffs when using Stochastic Gradient Descent vs the ADAM optimizer.** 

+ Video explaining the SGD optimizer (3 min)

+ Video explaning the ADAM optimizer (3 min)

+ Choosing an optimizer:
  + Training a net with Stochastic Gradient Descent (tf.train.GradientDescentOptimizer)
  + Traning a net with ADAM (tf.train.AdamOptimizer)

+ Video introduction to regularization and dropout (2-3 min)

**Lesson 4.4: Regularizing the network**

**A learning objective: use regularization with tf.nn.dropout to improve the perfomance of the network in test data**

+ Video explaining the need for regularization (3 min)

+ Quiz on regularization

+ Drill: Implement regularization with tf.nn.dropout
  
**Lesson 4.5:  Putting it all together**

**A learning objective: compare the performance tradeoffs of U-nets with different hyperparameters**

+ Video on the uses of U-net and how its parts fit together (3 min)

+ Evaluating the performance of the network with TensorBoard


