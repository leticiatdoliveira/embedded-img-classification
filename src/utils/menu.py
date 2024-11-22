#src/utils/menu.py
import os


class Menu:
    def __init__(self):
        self.options = ["Mobilenet", "Exit"]

    def add_option(self, option):
        if "Exit" in self.options:
            self.options.insert(-1, option)
        else:
            self.options.append(option)

    def show(self):
        os.system('clear')
        print("Menu:")
        for i, option in enumerate(self.options):
            print(f"{i + 1}. {option}")

    def get_choice(self):
        try:
            choice = int(input("Enter your choice: "))
            if choice < 1 or choice > len(self.options):
                raise ValueError
        except ValueError:
            print("Invalid choice. Enter a valid number.")
            return self.get_choice()
        return choice

