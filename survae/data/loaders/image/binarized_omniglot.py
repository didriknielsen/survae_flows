from survae.data.datasets.image import OMNIGLOTDataset
from torchvision.transforms import Compose
from survae.data.transforms import Flatten, DynamicBinarize
from survae.data import TrainTestLoader, DATA_PATH


class DynamicallyBinarizedOMNIGLOT(TrainTestLoader):
    """
    The OMNIGLOT dataset of
    (Lake et al., 2013): https://papers.nips.cc/paper/5128-one-shot-learning-by-inverting-a-compositional-causal-process
    as processed in
    (Burda et al., 2015): https://arxiv.org/abs/1509.00519
    using dynamic binarization.
    """

    def __init__(self, root=DATA_PATH, flatten=False):

        self.root = root

        # Define transformations
        trans = [DynamicBinarize()]
        if flatten: trans.append(flatten_transform)

        # Load data
        self.train = OMNIGLOTDataset(root, train=True, transform=Compose(trans))
        self.test = OMNIGLOTDataset(root, train=False, transform=Compose(trans))
