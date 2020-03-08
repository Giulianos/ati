import tkinter as tk

# Create App window
window = tk.Tk()

# Set window title
window.title('ATI')

# Test the library with a label
label = tk.Label(window, text = "Hello World!")
# Position the label on the window and add padding
label.pack(padx=20, pady=20)

# Start the app
window.mainloop()
