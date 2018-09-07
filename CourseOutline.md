# Course: Image Segmentation with Python

### Course developer and instructor: [Antonio Rueda-Toicen, MSc.](https://www.linkedin.com/in/antonioruedatoicen/)
+ Technical expert and data science mentor at [Thinkful](https://www.thinkful.com/)
+ Research programmer at the [Algorithmic Dynamics Lab](https://www.algorithmicdynamics.net/)
+ Instructor at the National Institute of Bioengineering (INABIO) [Universidad Central de Venezuela](http://www.ucv.ve/)
+ CTO at [The Chain](http://thechain.tech/)

emails (use only one): antonio.rueda.toicen@gmail.com, atoicen@thinkful.com, antonio.rueda@ucv.ve, antonio.rueda.toicen@algorithmicnaturelab.org

**homepage: www.digital-spaceti.me**

### Technology stack

+ PyTorch
+ Sckit-image
+ OpenCV
+ Google's Colaboratory

### Related courses

+ Intro to PyTorch (in the list of requested courses)
+ [Convolutional Neural Networks for Image Processing](https://www.datacamp.com/courses/convolutional-neural-networks-for-image-processing)

## Course Description

Image segmentation is the computer vision task of partitioning an image into visually salient or semantically significant portions. It has a wide variety of scientific and industrial applications. We will study some of these using both traditional image processing approaches and state-of-the art deep learning methods. We will work with OpenCV, scikit-image, and PyTorch. 


Deep learning methods use data to train neural network algorithms to do a variety of machine learning tasks, such as classification of different classes of objects. Convolutional neural networks are deep learning algorithms that are particularly powerful for analysis of images. This course will teach you how to construct, train and evaluate convolutional neural networks. You will also learn how to improve their ability to learn from data, and how to interpret the results of the training.


## Chapter 1 - Low level segmentation

![K-medoids brain clustering](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/seeded%20segmentation.PNG)
   * Lesson 1.1 - Segmenting biomedical images through clustering
     * A learning objective: Create and compare oversegmented brain tumor images in magnetic resonance with K-means, K-medoids, and SLIC.
        * Spectral clustering of images 
        * How the number of clusters affects the result
        * Oversegmentation as dimensionality reduction
   * Lesson 1.2 -  Measuring segmentation quality
     * A learning objective: evaluate the quality and tradeoofs of different segmentations of brain tumors through precision, recall, especificity, and the f1-measure
     
     ![Ground truth image](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/TumorSim%20ground%20truth.png =440x)
     
       * Comparing a segmentation against a ground truth
        * Building a confusion matrix: Counting true positives, false positives, true negatives, false negatives
        * Precision vs recall tradeoffs
        * The f1-measure (aka Dice Coefficient) as a metric of segmentation quality of brain tumors
   * Lesson 1.3 - Watershed and the GrowCut cellular automaton
     * A learning objective: segment multispectral satellite images from LANDSAT using unsupervised seed selection for GrowCut and the watershed transform algorithms.
     
     ![GrowCut](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/GrowCut.png)
        * Using manually defined segmentation seeds
        * Finding and clustering local entropy minima
        * Image gradients and the watershed transform
        * Adversarial evolution in the GrowCut cellular automaton
         ![3D GrowCut](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/3d-GrowCut-brain.PNG =350x)      
      * 3D GrowCut
          * Running GrowCut on a GPU: the stencil operation
          * Reducing oversegmentation with GrowCut
   
## Chapter 2 - Semantic Segmentation

   * Lesson 2.1 - Labeling images with Fuly Convolutional Neural Networks
     * A learning objective: Segment cats from images using deep neural networks
![Segment cat](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/segment%20cat.PNG)     
        * Brief intro to PyTorch
        * Units and layers in a neural network
        * The gradient descent algorithm
        * Backpropagation
        * Learning rate
        * Creating convolutional layers 
        * ReLU
        * The pooling operation, max pooling
        * Loss functions: multiclass cross entropy, quadratic loss
        * Using a repurposed Inception architecture for image segmentation
  
  
 ## Chapter 3 - Using encoder-decoder architectures for segmentation
   * Lesson 3.1 - Creating encoder-decoder architectures for segmentation
   
   ![segnet-roads](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/Segnet-roads.PNG)
     
     * A learning objective: Segment images of roads using Segnet.
        * Architecture of the VGG16 network
        * Using an encoder-decoder architecture
        * Batch normalization
        * Downsampling vs upsampling
        * Using skip connections
        * From encoding to decoding: reversing operations
        * Unpooling options
          * Bilinear interpolation
          * "Bed of nails"
          * Max location switches
        * The transposed convolution
        * The softmax function
  
  * Lesson 3.2 - U-net for biomedical semantic segmentation
  ![U-net architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png =450x)
      * A learning objective: Segment cell nuclei images from Kaggle's data science bowl using U-net
          * The U-net architecture
          * Skip connections with concatenations
          * Doubling the number of feature channels
          * Cropping layers
          
          
## Chapter 4 - Segmenting images with Generative Adversarial Networks

   * Lesson 3.1 - Using generative adversarial networks to segment scenery
     * A learning objective: create a GAN to segment and generate images.
     ![GAN](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/gan.PNG =360x)
        * Understanding GAN architecture
          * Generative vs Discriminative Networks 
        * Vanishing gradients
        * Exploding gradients
        * Minibatch stochastic gradient descent
        * ADAM 
        * Image to image translation: CycleGAN        
