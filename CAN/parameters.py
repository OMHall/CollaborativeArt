# Root directory for dataset
dataroot = './CAN/wikiart/wikiart_16_9/'
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 16
# smaller batchsize more stable training

# Number of styles
n_class = 27

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size_a = 256
image_size_b = 144

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0
