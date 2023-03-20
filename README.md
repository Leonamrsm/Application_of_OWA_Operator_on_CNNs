# Application_of_OWA_Operator_on_CNNs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains the aggregation layers implemented in the article ["Application of Learned OWA Operators in Pooling and Channel Aggregation Layers in Convolutional Neural Networks"](https://sol.sbc.org.br/index.php/eniac/article/view/22813) using keras/tensorflow.

The implemented layers can be found in the `layers` folder.

## OWAPooling

Ordered Weighted Average Pooling operation for spatial data. For each pooling region activations are sorted and weighted according to a trainable weights learned during training.

Developed based on the article ["Learning ordered pooling weights in image classification"](https://www.sciencedirect.com/science/article/abs/pii/S0925231220309991)

```
OWAPooling(pool_size=(2, 2), strides=None, padding=(0,0), name=None, sort=True, train=True, seed=None, all_channels=False)
```

##### Arguments

* **pool_size:** tuple of 2 integers. Window size on which the pooling operation will be performed.
* **strides:** tuple of 2 integers, or None. Strides values. If `None`, it will default to pool_size.
* **padding:**  One of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
* * **name:**  The name of the layer (string).
* **sort:** True or False. Defines whether the values of each pooling window will be sorted in descending order.
* **train:** True or False. True or False. Defines whether the weights of OWA operators will be trained. 
* **seed:** A Python integer. Used to make the behavior of the initializer deterministic. Note that a seeded initializer will produce the same random values across multiple calls.
* **all_channels:** True or False. If True 1 set of weights is used for each channel. Of False the same set of weights is used for all channels

## OWAConv_fm



```
OWAConv_fm(pool_size=(2, 2), strides=None, padding=(0,0), name=None, sort=True, train=True, seed=None, all_channels=False)
```

(tf.keras.layers.Layer):
    def __init__(self,
               filters, 
               strides=(1, 1),
               data_format=None,
               name=None,
               train=True, 
               seed=None,
               **kwargs):
