# Computer_Vision_Compilation
A compilation of computer vision projects, including video, object detection, segmentation, classification, autoencoders, steganography, ect.

So much computer vision so little time. My GitHub can get rather cluttered making a new repository to feature for each one and some smaller or older code samples need a good home without obscuring newer and more interesting projects.

# AI_Generating_Dance_Videos
Using Autoencoder CNN and Stacked LSTM RNN to train an computer to generate it's own dance videos

![alt text](https://github.com/PatrickBD/AI-Generating-Dance-Videos/blob/master/Dance_Robots_Comic.jpg)

This code is made to run on Kaggle Kernels. Fork the code here: https://www.kaggle.com/valkling/how-to-teach-an-ai-to-dance

Watch a sample output from this notebook here: https://youtu.be/1IvLdoXSoaU

Kaggle dataset: https://www.kaggle.com/valkling/shadow-dancers-videos
^(Go here for data that could not fit with Github's data limits)

### Teach an AI to Dance

NLP and image CNNs are all the rage right now, here we combine techniques from both to have a computer learn to make it's own dance videos. This notebook is a consolidation of 3 smaller notebooks I made for this project:

Part 1-Video Preprocessing: We will take the frames from a dance video of silhouettes, preprocess them to smaller and simpler, and add them to a zip file in sequence.

Part 2-Autoencoder Compression: To save even more memory for our model, we will compress these frames with an Autoencoder into a much smaller numpy array.

Part 3-Train AI w/ RNNs: We will put these compressed frames into sequences and train a model to create more.

I based this kernel off the project in this youtube video: https://www.youtube.com/watch?v=Sc7RiNgHHaE While he does not share his code, the steps expressed in the video were clear enough to piece together this project. Thanks to Kaggle's kernals GPU and some alterations, we can achieve even better results in less time than what is shown in the video. While still pretty computationally expensive for modern computing power, using these techniques for a dancing AI opens up the groundwork for AI to predict on and create all types of different videos.

### AI Generated Last Week Tonight Zebra

Last Week Tonight released a green screen video of a zebra dancing and doing various other activities for viewers to edit into their own videos. This video is actually pretty good for video processing algorithms and AI video training. Let's try using this to create our own AI generated zebra dancing video.

This code is made to run on Kaggle Kernels. Fork the code here: https://www.kaggle.com/valkling/last-week-tonight-ai-generated-zebra-color

Watch a sample output from this notebook here: https://youtu.be/_Eq-u67ZJRI

Kaggle Zebra dataset: https://www.kaggle.com/valkling/last-week-tonight-zebra-video
^(Go here for data that could not fit with Github's data limits)

# Autoencoder_Generated_Faces
Training an autoencoder model on faces to generate new faces and find key features using PCA

Original Kaggle Kernel here: https://www.kaggle.com/valkling/generating-new-faces-using-autoencoders

The LAG dataset used: https://www.kaggle.com/gasgallo/lag-dataset

# Data_Science_Bowl_2018
Object Detection of cells for Data Science Bowl 2018 on Kaggle

The Kaggle Data Science Bowl 2018 was about automatically detecting and separating cells in slide images from under a microscope. One of the most complex and difficult object detection problem, up to dozen of visually similar cells need to be seperated in a pixel by pixel basis. The competition ended in April of 2018

### U-Net

A U-Net is a CNN model that maintains the benefits of max pooling while retaining pixel location for accurate object detection. In a normal CNN structure, max pooling is usually used to to make the model understand that important features are not location dependant while shrinking down the image. However, this removes the location information entirely which is not good for object detection. A U-Net starts by following a similar CNN structure with max pooling but then expands out similar to an auto encoder. The difference is that U-Nets retain information from between pooling layers and feed that information when expanding back out (Deconvolutional NN). This way, U-Nets gain feature detection advantages and independence while retaining the exact pixel locations for object detection.

U-Nets were the state of the art go to model for this type of object detection only a few years ago, dominating similar competitions easily. While still very useful and accurate, some other models like Mask RCNN can get better results nowadays. 

The Kaggle competition and related dataset can be found here: https://www.kaggle.com/c/data-science-bowl-2018

# Dog_Breed_Identification

*You are provided a strictly canine subset of ImageNet in order to practice fine-grained image categorization. How well you can tell your Norfolk Terriers from your Norwich Terriers? With 120 breeds of dogs and a limited number training images per class, you might find the problem more, err, ruff than you anticipated.

This notebook classifisy an image of a dog into 120 possible breeds. A straight forward classifier using VGG19, I don't normally use pre-trained models these days as I like to play around with custom structures. At the time I did not have much access to GPU serveces so that ended up being a limiting factor on how much I could train and customize this model.

The Kaggle competition and related dataset can be found here: https://www.kaggle.com/c/dog-breed-identification

# Humpback_Whale_Identification_Challenge
Image Classification of whale tails to determine identify the whale it came from

The difficulty in this image classification challenge comes from having thousands of different whales to classify between, often with one image example of each. This is mitigated somewhat with the allowance of 5 predictions per test image. Another part of the challenge is dealing with whales that have not been seen yet, which unlike the other categories, has ~800 samples. This makes for an extreme local minimum that is very tricky to get out of. Recognising this is the first step to a decent predictive model.

I got to this challenge late and only had 2 days to work on and train it. Still made substantial improvements passed the baseline in that time and improved the model after the competition deadline.

The Kaggle competition and related dataset can be found here: https://www.kaggle.com/c/whale-categorization-playground

# Image_Colorization
Image Colorization Using Autoencoders and Resnet

This notebook is made to work on Kaggle and uses Kaggle datasets. See and fork the notebook here: https://www.kaggle.com/valkling/image-colorization-using-autoencoders-and-resnet

This notebook colorizes grayscale images using a fusion of autoencoders and Resnet neural networks. The data is trained on ~2000 classic art paintings that were converted to grayscale and attempted to recolor. The NN uses a Resnet classifier to identify objects in the images to get a sense of what things should be colored.

# Image_Steganography_in_Python

Image Steganography: Hiding images, text files, and other data inside of images.

## Hiding Star Wars in Images

Steganography is the practice of concealing a file, message, image, or video within another file, message, image, or video. Whereas cryptography is the practice of protecting the contents of a message alone, steganography is concerned with concealing the fact that a secret message is being sent as well as concealing the contents of the message. (https://en.wikipedia.org/wiki/Steganography) This notebook will use least significant bit steganography to hide 100s of KB to several MB of data in an image without perceptibly changing the image. This is my attempt to duplicate the method without looking up tutorials or code samples on least significant bit steganography. My only knowledge going into this is a basic description of the method.

### Least Significant Bit Image Steganography Explained

Here is a quick rundown of how this works. Each pixel in a color image has a value for each of it's 3 color channel. These values are between 0 and 255 corresponding to how red, green, or blue that pixel is. These values can be converted to an 8 bit binary value (255 = 11111111, ect.). While changing the left-most bits can mean a lot of change to the value, the rightmost bits mean very little. We are going use the rightmost 2 bits for our encoding. Changing these 2 bits will change the value for that one color channel on that one pixel by at most 3 points but more likely less or not at all. The result is a difference in the final image that is imperceptible to the human eye. If we are willing to replace these 2 bits of the 8 bit value, we can use up to 1/4th of the image's total space for whatever else we want! If we convert our scripts to 8 bit binary as well, there is more than enough space in a single color image to replace the last 2 bits of each color channel with our scripts.

(Even more bits could be used for encoding. This allows for using more of the image's total space but runs the risk of making the changes more visible. The last 2 bits is plenty for this notebook and most encodings. )

For example this image of the Death Star has all 3 of the original Star Wars Scripts encoded in it:

![alt text](https://github.com/PatrickBD/Image-Steganography-in-Python/blob/master/Death_Star_With_Scripts.jpg)

More impressively, this Death Star image has several layers of other images encoded in it like an Image Steganography nesting doll ending with the script for A New Hope (Death Star => X-Wing => R2D2 => Death Star Plans => New Hope Script)

![alt text](https://github.com/PatrickBD/Image-Steganography-in-Python/blob/master/Encoded_Death_Star_HD.jpg)

# MNIST Notebooks

Going to try other image processing techniques, might as well try it out on MNIST. An assortment of notebooks done on everyone's favorate image practice dataset.

![alt text](https://github.com/PatrickBD/MNIST/blob/master/why-not-mnist.jpg)

# Plant_Seedlings_Classification
My Notebook for Plant Seedlings Classification Image Processing Challenge on Kaggle

*Can you differentiate a weed from a crop seedling?*

*The ability to do so effectively can mean better crop yields and better stewardship of the environment.*

*The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset* *containing images of approximately 960 unique plants belonging to 12 species at several growth stages.*

I used this with some voting ensembling between model variations for my final score.

The Kaggle competition and related dataset can be found here: https://www.kaggle.com/c/plant-seedlings-classification
