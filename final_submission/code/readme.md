# Code Readme
simple_baseline.py contains 2 baselines of milestone2. 
Put all the data into the same folder, open an IDE and click 'run' to run the program.

Cis530_baseline.ipynb, Cis530_extension1.ipynb and Cis530_CNNLSTM_extension2.ipynb are the prublish baseline of milestone3 and
2 extensions of milestone4. 

Here's the code that needs to be changed.
```python
train_iterator, val_iterator, test_iterator, covid19_iterator = load_data('./gdrive/Shared drives/cis530_final_project/MS4/data/train.csv', './gdrive/Shared drives/cis530_final_project/MS4/data/val.csv', 
                                                          './gdrive/Shared drives/cis530_final_project/MS4/data/test.csv','./gdrive/Shared drives/cis530_final_project/MS4/data/covid19.csv')
```
It should be change to 
```python
train_iterator, val_iterator, test_iterator, covid19_iterator = load_data('your_path/train.csv', 'your_path/val.csv', 
                                                          'your_path/test.csv','your_path/covid19.csv')
```


Extension3 was to mannually annotate Covid19 tweets and apply the dataset as test to different models. 
5 models were applied and all the result could be found in the related python files 
(Cis530_extension1.ipynb,Cis530_CNNLSTM_extension2.ipynb,Cis530_baseline.ipynb,simple_baseline.py(2 models inside)).

pre.py is used to clean tweet data, and is no longer useful.

computePRF.py is used for evaluation and the instruction could be found in output/