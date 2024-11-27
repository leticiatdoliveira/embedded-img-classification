#src/utils/menu.py


def get_model_name(user_option: int) -> str:
    """
    Get the model name based on user option

    :param user_option:
    :return:
    """
    if user_option == 1:
        return "MobileNetV2"
    elif user_option == 2:
        return "ResNet18"
    else:
        return "YOLOv3"


def menu() -> str | None:
    """
    Create a menu to choose model type

    :return usr_choice: User input for model type
    """
    while True:
        print("Choose the model type:")
        print("1. MobileNetV2")
        print("2. ResNet18")
        print("3. YOLOv3")
        print("4. Exit")
        usr_choice = input("Enter the model type: ")

        if usr_choice.isdigit() and 1 <= int(usr_choice) <= 4:
            if int(usr_choice) == 4:
                print("Exiting the script")
                return None
            else:
                return get_model_name(int(usr_choice))
        else:
            print("Invalid choice. Please enter a valid choice.")
