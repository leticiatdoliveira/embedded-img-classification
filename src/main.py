# src/main.py
import cv2
from src.utils.camera import Camera
from src.utils.logger import Logger
from src.utils.menu import Menu
from src.models.mobilenetv2_model import MobileNetV2Model


def main():
    # Display menu
    menu = Menu()
    menu.show()
    model_type = menu.get_choice()
    if model_type == 2:
        exit()

    # Initialize logger
    logger = Logger(script_name="main_model={}".format(model_type))
    logger.log("Starting the object detection script")

    # Initialize camera
    camera = Camera()
    logger.log("Camera initialized")

    # Initialize model
    model = MobileNetV2Model()
    logger.log("MobileNetV2 model initialized")

    # real time loop to detect objects
    try:
        while True:
            frame = camera.read_frame()
            tensor = model.preprocess(frame)
            output = model.predict(tensor)
            top_predictions = model.get_top_predictions(output, top_k=2)
            logger.log(f"Top predictions: {top_predictions}")

            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.log(f"An error occurred: {e}")
    finally:
        camera.cap.release()
        cv2.destroyAllWindows()
        logger.log("Object detection script ended")


if __name__ == '__main__':
    main()