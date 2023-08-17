import cv2
from cv2 import dnn_superres


def upscale(img_path = None, scale = 8):
    # upscale times 4 or 8
    # !requires download pretrained models!


    image = cv2.imread(img_path)
    # for example: "./CAN/plots/99.png"

    if scale == 8:
        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()

        # Read the desired model
        model_path = "./upsampling_models/LapSRN_x8.pb"
        sr.readModel(model_path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("lapsrn", 8)
        
        # Upscale the image
        result = sr.upsample(image)


    else:
        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()

        # Read the desired model
        model_path = "./upsampling_models/EDSR_x4.pb"
        sr.readModel(model_path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("edsr", 4)

        # Upscale the image
        result = sr.upsample(image)

    # show the image
    #cv2.imshow("Original", image)
    #cv2.imshow("Result_Upscale", result)

    # Save the image
    cv2.imwrite(img_path.rsplit('.', 1)[0] + '_upscale.png', result)

    return result
