import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # Set window title
        self.title('ATI')

        # Test the library with a label (store as class property to change it later)
        self.label = tk.Label(self, text = 'Hello World!')
        # Position the label on the window and add padding
        self.label.pack(padx=20, pady=20)

        # Add a button and set `on_bye` as click handler
        button = tk.Button(self, text='Bye', command=self.on_bye)
        # Position the button
        button.pack()


    def on_bye(self):
        # Update the label
        self.label.configure(text = 'Bye World!')
    
