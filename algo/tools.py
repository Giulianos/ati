class Tools():
    def __init__(self, app_ref):
        self.app_ref = app_ref

    def test_selection(self):
       print('Requested selection') 
       # Here we request a selection that will be handled
       # by test_selection_handler
       self.app_ref.mouse_selection.request_selection(self.test_selection_handler)

    # This is called once we have the selection
    def test_selection_handler(self, start, end):
       # Here we do something with the selection
       print('Successfully selected from {} to {}!'.format(start, end))
