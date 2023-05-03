# Learning based Image Coloring using General Adversarial Networks and Convolutional Neural Network
The project developed an end-to-end system that can colorize grayscale images effectively using a deep neural network. We leverage the General Adversarial Networks, wherein two models, the generative and discriminative models are trained simultaneously. In this method, a grayscale image is fed into a Generative Adversarial Network (GAN), which generates a colored rendition of the original image. The training procedure will use a Discriminative network to distinguish between real and false images, while the GAN will attempt to trick the Discriminative network by producing colors that provide the illusion of authenticity. We want to construct a stable and accurate system for automated grayscale image colorization using this combination approach.

### Some results of our model on COCO dataset
<img src="results/r1.png" width = "750" >
<img src="results/r2.png" width = "750">
  
 Top Row - Input Grayscale Image  
 Middle Row - Output as per our model  
 Bottom Row - Ground Truth

## Instructions to Run the program
### Step1: Install the dependencies
Go to the directory containing requirements.txt 
```bash
pip install -r requirements.txt
```

### Step2: You can either train the model from scratch or use our trained model to test the results.
#### Part A: Using the existing model
Go to the materials directory, you will find a text file containing a drive link. The drive link contains our pretrained model. Download the zip file and extract them to the location where the file crayons.py exists.

After that, load any grayscale image into the directory having crayons.py and simple run the program using
```bash
python3 crayons.py
```

#### Part B: Training from scratch
Go the directory having train.py and simply run the program using
```bash
python3 train.py
```
**Note**: Running train.py can take time. In our case, it took 10-11 hours when trained on Single Nvidia A100 GPU 

## References
1. Isola, P., Zhu, J., Zhou, T., Efros, A. A. (2016). Image-to-Image Translation with Conditional Adversarial Networks. ArXiv. /abs/1611.07004
2. https://github.com/mberkay0/image-colorization

## Authors
Varun Sreedhar, Jignesh Kirti Nagda, and Venkhatesh Arunachalam
