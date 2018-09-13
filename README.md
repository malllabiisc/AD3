# AD3
## AD3: Attentive Deep Document Dater

Source code and dataset for [EMNLP 2018](http://emnlp2018.org) paper: [AD3: Attentive Deep Document Dater](http://malllabiisc.github.io/publications/).

![](https://github.com/malllabiisc/AD3/blob/master/Model.png)
*Overview of AD3 (proposed method), an attention-based neural document dating system which utilizes both context and temporal information in documents in a flexible and principled manner. Please refer paper for more details.*

### Dependencies

* Compatible with TensorFlow 1.x and Python 3.x.
* Dependencies can be installed using `requirements.txt`.

### Dataset:

* Download the processed version (includes dependency and temporal graphs of each document) of [NYT](https://drive.google.com/file/d/1wqQRFeA1ESAOJqrwUNakfa77n_S9cmBi/view?usp=sharing) and [APW](https://drive.google.com/open?id=1tll04ZBooB3Mohm6It-v8MBcjMCC3Y1w) datasets.
* Unzip the `.pkl` file in `data` directory.
* Documents are originally taken from NYT and APW section of [Gigaword Corpus, 5th ed](https://catalog.ldc.upenn.edu/ldc2011t07).

### Preprocessing:

For getting temporal graph of new documents. The following steps need to be followed:

- Setup [CAEVO](https://github.com/nchambers/caevo) and [CATENA](https://github.com/paramitamirza/CATENA) as explained in their respective repositories.

- For extracting event and time mentions of a document

  - `./runcaevoraw.sh <path_of_document>`

  - Above command generates an `.xml` file. This is used by CATENA for extracting temporal graph and it also contains the dependency parse information of the document which can be extracted using the following command:

    ```shell
    python preprocess/read_caveo_out.py <caevo_out_path> <destination_path>
    ```

- For making the generated `.xml` file compatible for input to CATENA, use the following script as

  ```shell
  python preprocess/make_catena_input.py <caevo_out_path> <destination_path>
  ```

- `.xml` generated above is given as input to CATENA for getting the temporal graph of the document. 

   ```shell
    java -Xmx6G -jar ./target/CATENA-1.0.3.jar -i <path_to_xml> \
    	--tlinks ./data/TempEval3.TLINK.txt \
    	--clinks ./data/Causal-TimeBank.CLINK.txt \
    	-l ./models/CoNLL2009-ST-English-ALL.anna-3.3.lemmatizer.model \
    	-g ./models/CoNLL2009-ST-English-ALL.anna-3.3.postagger.model \
    	-p ./models/CoNLL2009-ST-English-ALL.anna-3.3.parser.model \
    	-x ./tools/TextPro2.0/ -d ./models/catena-event-dct.model \
    	-t ./models/catena-event-timex.model \
    	-e ./models/catena-event-event.model 
    	-c ./models/catena-causal-event-event.model > <destination_path>
   ```

   The above command outputs the list of links in the temporal graph which are given as input to AD3. The output file can be read using the following command:

   ```shell
   python preprocess/read_catena_out.py <catena_out_path> <destination_path>
   ```

    

### Usage:

* After installing python dependencies from `requirements.txt`, execute `sh setup.sh` for downloading GloVe embeddings.

* `ad3.py` contains TensorFlow (1.x) based implementation of AD3 (proposed method). 
* To start training: 
  ```shell
  python ad3.py -data data/nyt_processed_data.pkl -class 10 -name test_run
  ```

  * `-class` denotes the number of classes in datasets,  `10` for NYT and `16` for APW.
  * `-name` is arbitrary name for the run.
