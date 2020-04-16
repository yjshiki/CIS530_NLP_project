# Evaluation Script

##Description:

The evaluation script takes in a “goldfile” and a prediction file to compute precision, recall and f1 score for a three-class multinomial classification. A confusion matrix is first computed. It is then used to calculate precision, recall and thus f1 for all 3 classes (positive, neutral and positive). In order to derive a single metric that shows the goodness of model, macroaveraging is used to compute averages over classes. Compared with microaveraging, macroaveraging better balances between classes and reflects the statistics of smaller classes. Since performance for each class is of same importance in our project, macroaveraging is more appropriate.

##Example

To run it from the command line, use following command and replace “train.csv” and “prediction.csv” with paths to your target files:

computePRF.py --goldfile train.csv --predfile prediction.csv

It prints out output including:
- confusion matrix,
- precision, recall and f1 for each class, 
- and macroaverages. 

##Reference

https://web.stanford.edu/~jurafsky/slp3/4.pdf