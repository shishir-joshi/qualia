# Big JSON file 
import json
from utils import SAMPLE_DATASET

class SampleDataset(object):

    def __init__(self, useful_keys=None):
        self.data = json.load(open(SAMPLE_DATASET))
        # Keeping just the title and summary as default
        self.useful_keys = useful_keys if not useful_keys is None else ['title', 'summary']

    def load(self):
        # Get it in line
        def join_ds(x, useful_keys):
            return '. '.join([x[useful_keys[0]], x[useful_keys[1]]])

        self.data = list(
                    map(lambda x: join_ds(x, self.useful_keys), 
                        self.data
                        )
                    )
        
        # Make numeric labels that correspond to the individual documents 
        self.data_labels = list(range(len(self.data)))

        # Make Label to document mapping for pretty outputs
        self.label_mapping = dict(
                            zip(self.data_labels, 
                                self.data
                                )
                            )

    def __getitem__(self, i):
        return self.data[i], self.data_labels[i]