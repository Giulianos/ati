import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # Set window title
        self.title('ATI')

        # Test the library with a label
        label = tk.Label(self, text = 'Hello World!')
        # Position the label on the window and add padding
        label.pack(padx=20, pady=20)

    
