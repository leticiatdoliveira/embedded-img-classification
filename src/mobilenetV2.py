import time
import torch
import numpy as np
from torch.jit import RecursiveScriptModule
from torchvision import models, transforms
import cv2
import logging
import os

# constants
IMG_SIZE = 224
FPS_VIDEO_READING = 36
FPS_MODEL = 30

# set the backend for quantized engine
torch.backends.quantized.engine = 'qnnpack'


def init_logging():
    """
    Initialize logging
    """

    # get the current file name
    script_name = os.path.basename(__file__)
    script_name = script_name.split(".")[0]

    # set where to save the log file
    log_file = f"logs/{script_name}.log"
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # set the logging format
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.info("Starting the script")


def set_nb_of_threads(nb_of_non_cpu: int = 2):
    """
    set the number of threads for the script
    
    :param nb_of_non_cpu: int - number of cpu to not use
    """
    # get the number of threads
    nb_cpu = os.cpu_count()
    logging.info(f"Number of CPUs available: {nb_cpu}")

    # set the number of threads
    nb_threads_to_use = nb_cpu - nb_of_non_cpu
    torch.set_num_threads(nb_threads_to_use)


def set_cam_capture() -> cv2.VideoCapture:
    """
    Set the camera capture

    :return cap: cv2.VideoCapture object
    """

    cap = cv2.VideoCapture(0)

    # set the frame rate (fps)
    cap.set(cv2.CAP_PROP_FPS, FPS_VIDEO_READING)

    # set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)
    return cap


def frames_preprocessing(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess the frame

    :param image: np.ndarray
    :return img_processed: torch.Tensor
    """

    # define the transformation
    preprocess = transforms.Compose([
        # transform image to CHW tensor format
        transforms.ToTensor(),

        # normalize colors to the range expected by mobilenetv2
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_processed = preprocess(image)

    return img_processed


def create_mobilenetv2_model() -> RecursiveScriptModule:
    """
    create the mobilenetv2 model

    :return scripted_net: RecursiveScriptModule
    """
    # load the trained model
    net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)

    # reduce Python overhead and fuse any ops (from ~20fps to ~30fps)
    scripted_net = torch.jit.script(net)

    return scripted_net


def get_two_highest_prediction(output: torch.Tensor) -> (int, int):
    """
    get the two highest predictions

    :param output: torch.Tensor
    :return: (int, int)
    """
    # apply softmax to the output
    labels_prob = torch.nn.functional.softmax(output[0], dim=0)

    # sort the output from the highest to the lowest
    labels_prob.sort(descending=True)

    # get the two highest predictions
    label_1 = labels_prob.indices[0].item()
    label_2 = labels_prob.indices[1].item

    return label_1, label_2


def main():
    # init logging
    init_logging()

    # set the number of threads
    set_nb_of_threads()

    # config camera capture
    cap = set_cam_capture()

    # create the model
    model = create_mobilenetv2_model()

    # start real time object detection
    last_log = time.time()
    frame_count = 0
    with torch.no_grad():
        # read the frames
        ret, image = cap.read()

        # check if the frame is read
        if not ret:
            logging.error("Error reading the frame")
            return

        # convert openCV BGR format to RGB format
        image = image[:, :, [2, 1, 0]]

        # preprocess the frame
        img_processed = frames_preprocessing(image)

        # reformat the image frame in minibatch
        img_batch = img_processed.unsqueeze(0)

        # run the model inference
        output = model(img_batch)

        # get 2 class prediction
        label_1, label_2 = get_two_highest_prediction(output)
        logging.info(f"Predictions: {label_1}, {label_2}")

        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_log >= 1:
            logging.info(f"FPS: {frame_count / (now - last_log)}")
            last_log = now
            frame_count = 0


if __name__ == "__main__":
    main()
