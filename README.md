# Learning based Image Coloring using General Adversarial Networks and Convolutional Neural Networks
The project developed an end-to-end system that effectively uses a deep neural network to colorize grayscale images. We leverage the General Adversarial Networks, wherein two generative and discriminative models are trained simultaneously. In this method, a grayscale image is fed into a Generative Adversarial Network (GAN), which generates a colored rendition of the original image. The training procedure will use a Discriminative network to distinguish between real and false images. At the same time, the GAN will attempt to trick the Discriminative network by producing colors that provide the illusion of authenticity. Using this combination approach, we want to construct a stable and accurate system for automated grayscale image colorization.

### Some results of our model on the COCO dataset
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

### Step2: You can train the model from scratch or use our trained model to test the results.
#### Part A: Using the existing model
You can visit the materials directory and find a text file with a drive link. The drive link contains our pre-trained model. Download the zip file and extract them to the location where the file crayons.py exists.

After that, load any grayscale image into the directory having crayons.py and run the program using
```bash
python3 crayon.py
```

#### Part B: Training from scratch
Go to the directory having train.py and run the program using
```bash
python3 train.py
```
**Note**: Running train.py can take time. In our case, it took approximately 20 hours when trained on a Single Nvidia A100 GPU 

## References
1. Isola, P., Zhu, J., Zhou, T., Efros, A. A. (2016). Image-to-Image Translation with Conditional Adversarial Networks. ArXiv. /abs/1611.07004
2. https://github.com/mberkay0/image-colorization

## Authors
Varun Sreedhar, Jignesh Kirti Nagda, and Venkhatesh Arunachalam
