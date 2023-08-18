# Collaborative Evolution of Art in the Latent Space of a Deep Generative Model

This project was created as a master thesis at the VU Amsterdam. It contains a Creative Adversarial Network (CAN) for the generation of artworks. Resolutions of 64 x 64 and 256 x 144 (16:9) are available. The models are implemented with PyTorch.

It further contains an evolutionary computing approach to control the latent variables from which the CAN's generator creates images. For this, the evolutionary algorithm framework [DEAP](https://github.com/deap/deap) was used. A local search following the work of Roziere et al. (2020) was implemented as mutation. To determine the quality of the art images, an automatic image evaluation metric based on Neural Image Assesment (NIMA) trained on Aesthetic Visual Analysis dataset (AVA) was applied. In addition, an interactive (collaborative) evaluation was implemented. The images were upsampled using The Laplacian Pyramid Super-Resolution Network (LapSRN).

## Examples

<img src="https://github.com/OMHall/CollaborativeArt/blob/main/Examples/Example_1.png" height=100% width=100%>
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>

## WikiArt Dataset

The publicly available WikiArts dataset was resized and used as training data. However, the models can also be trained on any other dataset.
The wikiarts dataset can be [downloaded from this repository](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

## References

(1) Creative Adversarial Network

[Creative Adversarial Network (CAN)](https://arxiv.org/pdf/1706.07068.pdf): Elgammal, A., Liu, B., Elhoseiny, M., Mazzone, M.: CAN: Creative adversarial networks, generating ”art” by learning about styles and deviating from style norms. arXiv:1706.07068 (2017)

The implementation of the CAN was based on [this repository](https://github.com/otepencelik/GAN-Artwork-Generation) and also inspired by [this](https://github.com/IliaZenkov/DCGAN-Rectangular-GANHacks2/tree/main). 

(2) Collaborative Evolution

Neural Image Assesment (NIMA): Talebi, H., Milanfar, P.: NIMA: Neural image assessment. IEEE Transactions on Image Processing 27(8), 3998–4011 (2018)

Aesthetic Visual Analysis (AVA): Murray, N., Marchesotti, L., Perronnin, F.: AVA: A large-scale database for aesthetic visual analysis. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 2408–2415 (2012)

[This implementation](https://github.com/titu1994/neural-image-assessment) was used for NIMA without changes (the Inception ResNet v2 model, [weights can be dowloaded here](https://github.com/titu1994/neural-image-assessment/releases/tag/v0.5)).

Laplacian Pyramid Super-Resolution Network (LapSRN): Lai, W.S., Huang, J.B., Ahuja, N., Yang, M.H.: Deep laplacian pyramid networks for fast and accurate super-resolution. In: IEEE Conferene on Computer Vision and Pattern Recognition (2017)

Upsampling was done using [OpenCV](https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres), the pretrained model can be [downloaded here](https://github.com/fannymonori/TF-LAPSRN).

Local Search: Roziere, B., Teytaud, F., Hosu, V., Lin, H., Rapin, J., Zameshina, M., Teytaud, O.: EvolGAN: Evolutionary generative adversarial networks. In: Proceedings of the Asian Conference on Computer Vision (2020)

## Usage

To train the GAN, the use of GPU is recommended, especially for the higher resolution. As soon as a trained generator is provided, CPU is sufficient. The requirement.txt file contains all requirements (PyTorch is thereby only installed for CPU though).

Everything concerning CAN training can be found in the corresponding folder. In parameter.py, the overall parameters can bet set. train.py expects a folder containing a training dataset already in the appropriate size. The notebook offers an analysis of the training results.

For the evolutionary computing approach, there are several notebooks whose names are descriptive. All of them require a trained generator, most of them further require the availability of a pretrained NIMA model and a pretrained LapSRN model.
