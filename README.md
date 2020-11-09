# [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)


## TODO
- [ ] Refactor source code to use with pytorch
    - [x] Default train
        - [x] Add loss
        - [x] Add metric
    - [ ] Pretrain 1292 class
        - [ ] CrossEntropy loss
        - [ ] Triplet Loss 
    - [x] Multi GPU trainning
    - [ ] Transform - Data augment
        - [x] Add cutout
        - [ ] Add cutmix
        - [ ] Add AutoAugment
    - [ ] Inference code

- [ ] Tensorflow 2.0 version
    - [ ] Default train
        - [ ] Build model
        - [ ] Data loader
        - [ ] Default transform (resize, normalize)
        - [ ] Add loss
        - [ ] Add metric
    - [ ] Multi GPU training
    - [ ] Transform - Data augment
        - [x] Add cutout
        - [ ] Add cutmix
        - [ ] Add AutoAugment
    - [ ] Inference code

