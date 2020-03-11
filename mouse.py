class MouseSelection():
    def __init__(self):
        self.start = None
        self.handler = None
        self.enabled = False

    # Call this when you need a selection
    # when the user finishes the selection,
    # handler is called with start and end
    # coordinates.
    def request_selection(self, handler):
        self.enabled = True
        self.handler = handler

    def start_drag(self, coord):
        self.start = coord

    def end_drag(self, coord):
        start = self.start
        end = coord
        self.start = False
        self.enabled = False
        self.handler(start, end)

    def get_start(self):
        return self.start

    def is_enabled(self):
        return self.enabled
