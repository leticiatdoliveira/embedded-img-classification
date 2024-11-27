# src/main.py
import cv2

from src.models.model import Model
from src.utils.camera import Camera
from src.utils.logger import Logger
import src.utils.menu as menu
import src.utils.preprocessing as preprocessing


def main():
    # User menu to choose model type
    model_type = menu.menu()
    if model_type is None:
        print("Exiting the script")
        exit(0)

    # Initialize logger
    logger = Logger(script_name="main_model={}".format(model_type))
    logger.log("Starting the object detection script")

    # Initialize camera
    camera = Camera()
    logger.log("Camera initialized")

    # Create model
    model = Model(model_type=model_type, apply_quantize=True, apply_jit=True)
    logger.log("{} model initialized (Quantize = {} and JIT = {})".format(model.model_name.capitalize(),
                                                                          model.quantized,
                                                                          model.jit))

    # Real time loop to detect objects
    try:
        while True:
            # read camera
            frame = camera.read_frame()

            # format the image to model input format
            image = preprocessing.convert_bgr_to_rgb(frame)
            image_tensor = preprocessing.normalize_image(image)
            image_batch = image_tensor.unsqueeze(0)

            # predict the image by the model
            output = model.predict(image_batch)
            top_predictions = model.get_top_predictions(output, top_k=2)
            logger.log(f"Top predictions: {top_predictions}")

            cv2.imshow('Object Detection', frame)

            # force breaking of while true loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.log(f"An error occurred: {e}")
    finally:
        # Release the camera
        camera.cap.release()

        # Close all windows
        cv2.destroyAllWindows()
        logger.log("Object detection script ended")


if __name__ == '__main__':
    main()