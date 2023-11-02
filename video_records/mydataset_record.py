from .video_record import VideoRecord

class MyDataset_VideoRecord(VideoRecord):

    def __init__(self, row):
        self.data = row

    @property
    def path(self):
        return self.data[0]

    @property
    def num_frames(self):
        return {'RGB': int(self.data[1]),
                'Flow': int(self.data[1])-1,
                'Gyro': int(self.data[2]),
                'Acce': int(self.data[2])}

    @property
    def label(self):
        return int(self.data[3])
