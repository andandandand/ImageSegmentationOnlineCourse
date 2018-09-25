# Course: Image Segmentation with Python

### Course developer and instructor: [Antonio Rueda-Toicen, MSc.](https://www.linkedin.com/in/antonioruedatoicen/)
+ Technical expert and data science mentor at [Thinkful](https://www.thinkful.com/)
+ Research programmer at the [Algorithmic Dynamics Lab](https://www.algorithmicdynamics.net/)
+ Instructor at the National Institute of Bioengineering (INABIO) [Universidad Central de Venezuela](http://www.ucv.ve/)
+ Chief Technology Officer at [The Chain](http://thechain.tech/)

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





## Part 1: Image Segmentation Basics

+ Intro video explaining what image segmentation is (5-10 min)

   ![Ground truth image](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/TumorSim%20ground%20truth.png)

+ Loading an image with OpenCV
  + Importing medical image data with PyDICOM
+ Using a mask to select regions of an image
  * Select the tumor region in brain MRI
  * Visualizing the tumor region in 3D
    ![3D GrowCut](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/3d-GrowCut-brain.PNG) 
    
+ Quick multiple choice quiz about image segmentation (purpose of segmentation, use of masks)
 
 
 + Intro video explaining segmentation with the GrowCut cellular automaton (5-10 min)
 
 + Seeded segmentation with the GrowCut cellular automaton
    + Complete a Numpy implementation of the GrowCut cellular automaton
    + Using entropy minima as segmentation seeds
    + Labeling entropy minima with k-medoids
    + Defining minima area for segments to avoid oversegmentation
    ![K-medoids brain clustering](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/seeded%20segmentation.PNG)

+ Evaluating the quality of a segmentation
  + Counting true positives, true negatives, false positives, and false negatives 
  - Computing precision
  + Computing recall
  + Computing the f1-measure
  + Computing Intersection over Union (IoU)
  + Computing the warping error
  + Computing the Rand error 


## Part 2: Understanding U-net part 1
**Segmenting biomedical images**

+ Intro video explaining what U-net is and how does deep learning enables automatic image segmentation (5-10 min)
  * *The intro video mentions the encoder-decoder architecture of U-net and how part 2 focuses on the encoder, part 3 on the decoder, and part 4 on the final layers of U-net and its optimization*

![texto alternativo](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

+ U-net architecture
  + Quick multiple choice quiz about U-net's architecture (purpose of U-net, contents of layers, what convolution is, what upsampling is)

![image convolution](https://i.stack.imgur.com/YDusp.png)
+ Understanding image convolutions
  + Applying a convolution filter
  + Changing the weights of a convolution filter

+ Video explaining the need of image cropping and data augmentation in U-net (5 min)

+ Doing image cropping

+ Doing data augmentation
  + Mirroring
  + Rotation
  + Vertical/horizontal flips 
  + Deformations

+ Video explaining the use of pooling (3 min)
  
+ Understanding pooling
  * Implement max pooling
  
+ Video explaining activation functions in U-net (3 min)  
  
![](https://cdn-images-1.medium.com/max/1600/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg)  
+ Understanding ReLU 
  * Implement Regular ReLU
  * Implement Leaky ReLU


## Part 3: Understanding U-net part 2
**Segmenting road images**

![](https://github.com/aschneuw/road-segmentation-unet/blob/master/report/figures/submission_selection/images_003.png?raw=true)

* Video explaining the decoder part of the U-net learning architecture, explains upsampling, mentions what unpooling and transposed convolutions are and their purpose (5-10 min)

+ Understanding unpooling 
  + Replacing each entry with an $𝑛 \times 𝑛$ matrix filled with the original entry (NumPy drill).
  + Replacing each entry with an $𝑛 \times 𝑛$ matrix with the original entry in the upper left and the other squares set to 0. [1506.02753]   (NumPy drill)
    
![Transposed convolution](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_3/Deconv_exp.PNG)  
  + Computing a transposed convolution with tf.nn.conv2d_transpose

## Part 4: Understanding U-net part 3
**Segmenting multispectral satellite images**

![](https://camo.githubusercontent.com/28891f9cc505787a642b1ddabdc162e236c26204/68747470733a2f2f692e696d6775722e636f6d2f686a49546670632e706e67)

Understanding the final layers and the loss function

+ Video explaining loss functions (5 min)

+ Quiz on loss functions 

+ Understanding loss functions:
  + Computing Cross entropy
  + Computing Mean squared error
  + Computing Mean absolute error
  + Using the Dice similarity coefficient as an error metric

+ Video explaining optimizers (5 min)

+ Choosing an optimizer:
  + Training a net with Stochastic Gradient Descent (tf.train.GradientDescentOptimizer)
  + Traning a net with ADAM (tf.train.AdamOptimizer)

+ Video introduction to regularization and dropout (2-3 min)

+ Regularization with Dropout
  + Regularization with tf.nn.dropout

+ Video introduction to mathematical morphology operations (2-3 min)

+ Mathematical morphology to define borders of the final segmentation mask 
  + Eroding an image
  + Dilation an image
  + Finding borders on image by the composition of dilation and erosion
  
+ Putting it all together
  + Comparing the performance of networks with different hyperparameters










