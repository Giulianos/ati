import tkinter as tk
from PIL import ImageTk

class ImageViewer(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, bd=2, relief=tk.SUNKEN)

        # Create image scrolling frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        xscroll = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=tk.E+tk.W)
        yscroll = tk.Scrollbar(self)
        yscroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.canvas = tk.Canvas(self, width=500, height=500, bd=0, highlightthickness=0, relief='ridge', xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        xscroll.config(command=self.canvas.xview)
        yscroll.config(command=self.canvas.yview)

        # Selection rectangle
        self.selection_rect = None
        
        # Bind mouse event handler
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Button-1>', self.on_mouse_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)

        self.mouse_handler = None
        self.mouse_selection = None

    def set_image(self, pil_img):
        # Create the image in the canvas
        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0,0, anchor='nw', image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def get_mouse_pos(self, event):
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        return (x,y)

    def set_mouse_handler(self, mouse_handler):
        self.mouse_handler = mouse_handler

    def set_mouse_leave_handler(self, mouse_leave_handler):
        self.canvas.bind('<Leave>', mouse_leave_handler)

    def on_mouse_move(self, event):
        self.canvas.configure(cursor='crosshair')

        if self.mouse_handler != None:
            mouse_pos = self.get_mouse_pos(event)
            self.mouse_handler(self, mouse_pos[0], mouse_pos[1])

        # Show selection box if it's enabled
        try:
            startx, starty = self.mouse_selection.get_start()
            endx, endy = mouse_pos

            if self.selection_rect == None:
                self.selection_rect = self.canvas.create_rectangle(
                        startx, starty, endx, endy,
                        outline='red')
            else:
                self.canvas.coords(self.selection_rect, startx, starty, endx, endy)
        except:
            pass

    def set_mouse_selection(self, mouse_selection):
        self.mouse_selection = mouse_selection

    # This is called when drag starts
    def on_mouse_press(self, event):
        if self.mouse_selection != None and self.mouse_selection.is_enabled():
           mouse_pos = self.get_mouse_pos(event)
           self.mouse_selection.start_drag(mouse_pos)

    # This is called when drags ends
    def on_mouse_release(self, event):
        if self.mouse_selection != None and self.mouse_selection.is_enabled():
            mouse_pos = self.get_mouse_pos(event)
            self.mouse_selection.end_drag(mouse_pos)
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
