class TrackedObject:
    def __init__(self, x, y, bb_x_half, bb_y_half):
        self.x = x
        self.y = y
        self.bb_x_half = bb_x_half
        self.bb_y_half = bb_y_half
        self.frames = []

    # returns a bb form upper left to bottom right->
    def get_bb(self):
        return ((self.x - self.bb_x_half, self.y - self.bb_y_half),
                (self.x + self.bb_x_half, self.y - self.bb_y_half),
                (self.x - self.bb_x_half, self.y + self.bb_y_half),
                (self.x + self.bb_x_half, self.y + self.bb_y_half))

    def get_color_bb(self):
        bb = self.get_bb()
        return ((bb[0][0] * 2, bb[0][1] * 2), (bb[1][0] * 2, bb[1][1] * 2),
                (bb[2][0] * 2, bb[2][1] * 2), (bb[3][0] * 2, bb[3][1] * 2))

    # returns new TR_object with difrences between the objects representing current velocity.
    def get_velocity_object(self, tr_object_new):
        return TrackedObject(tr_object_new.x - self.x,
                             tr_object_new.y - self.y,
                             tr_object_new.bb_x_half - self.bb_x_half,
                             tr_object_new.bb_y_half - self.bb_y_half)
