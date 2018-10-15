# AD3
## AD3: Attentive Deep Document Dater

Source code and dataset for [EMNLP 2018](http://emnlp2018.org) paper: [AD3: Attentive Deep Document Dater](http://malllabiisc.github.io/publications/).

![](https://github.com/malllabiisc/AD3/blob/master/Model.png)
*Overview of AD3 (proposed method), an attention-based neural document dating system which utilizes both context and temporal information in documents in a flexible and principled manner. Please refer paper for more details.*

### Dependencies

* Compatible with TensorFlow 1.x and Python 3.x.
* Dependencies can be installed using `requirements.txt`.

### Dataset:

* We evaluate AD3 on NYT and APW section of [Gigaword Corpus, 5th ed](https://catalog.ldc.upenn.edu/ldc2011t07). For preprocessing refer [NeuralDater](https://github.com/malllabiisc/NeuralDater).

### Usage:

* After installing python dependencies from `requirements.txt`, execute `sh setup.sh` for downloading GloVe embeddings.

* `ac_gcn.py` and `oe_gcn.py` contains TensorFlow (1.x) based implementation of AD3 (proposed method). 
* To start training: 
  ```shell
  python ac_gcn.py -data data/nyt_processed_data.pkl -class 10 -name test_run
  python oe_gcn.py -data data/nyt_processed_data.pkl -class 10 -name test_run
  ```

  * `-class` denotes the number of classes in datasets,  `10` for NYT and `16` for APW.
  * `-name` is arbitrary name for the run.
