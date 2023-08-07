from mmseg import registry
from mmseg.datasets.basesegdataset import BaseSegDataset

@registry.DATASETS.register_module()
class DeepCrackDataset(BaseSegDataset):
    """DeepCrack Dataset.
    To use DeepCrack Dataset, it is required to convert the label format 
    from 0, 255 to 0, 1.
    
    References:
        [1] DeepCrack: A Deep Hierarchical Feature Learning Architecture 
            for Crack Segmentation (https://github.com/yhlleo/DeepCrack)
        [2] Preprocessing for DeepCrack Dataset 
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