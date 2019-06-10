# Classic-CNN-Model-PyTorch
reimplement classic CNN model and test on CIFAR-10 dataset

## Prerequisites

   * python 3.5+
   * pytorch 1.0+
   
## License

This project is released under the [MIT license](LICENSE)

## Getting Started

Please Choose the model you would like to use at ``train.py``, and using the model
defined in the ``<classic-CNN-model dir>/models`` replace the variable ``model`` in the train.py
after you choose the model, just use

```
python train.py --lr <learning rate> --resume <resume_path> --gpu <whether use gpu> --work_dir <model save directory> --max_epoch<max_epoch> 
```

can start the training process



   