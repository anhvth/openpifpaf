from .preprocess import Preprocess


class Encoders(Preprocess):
    """Preprocess operation that runs encoders."""
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, image, anns, meta):
        encoded_anns = []
        for i, enc in enumerate(self.encoders):
            # create coded annotations for each head
            encoded_anns += [enc(image, anns, meta)]
        meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        return image, encoded_anns, meta
