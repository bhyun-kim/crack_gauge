from mmseg import registry
from mmseg.datasets.basesegdataset import BaseSegDataset

@registry.DATASETS.register_module()
class CRACK500Dataset(BaseSegDataset):
    """CRACK500 Dataset.
    To use CRACK500 Dataset, it is required to convert the label format 
    from 0, 255 to 0, 1. We use the full size (around 2000x1500) images
    and labels for training
    
    References:
        [1] Feature Pyramid and Hierarchical Boosting Network for Pavement
            Crack Detection (https://arxiv.org/abs/1901.06340)
        [2] Preprocessing for CRACK500 Dataset 
            (https://github.com/bhyun-kim/prep_deepcrack)
    """
    
    METAINFO = dict(classes=('background', 'crack'),
                    palette=[[0, 0, 0], [0, 0, 255]])
    
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_binary_label.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)