# Baseline 1
### Description
The first is ramdom_baseline. For each text, randomly lable as -1=negative,0=neutral,1=positive.

### Sample Output
```python
print(train_rand_pred[0:10])
```
[0, 0, 0, 1, -1, -1, 1, 0, 0, 0]

### Test Result
Precision, recall,f1-score were used for evaluation. The test result using evaluation metric is

|            |positive    |negative    |neutral     |macroavg 
|:-----------:|-----------------|:------------:|:-----:|:-------:|
|precision   |0.3136412870340873|0.15163425209828335|0.5339445341607009|0.33307335776435715
|recall      |0.3321861419278214|0.33337979094076653|0.33320108234211215|0.33292233840356666
|f1 score          |0.32264745787152044|0.20845497564248927|0.41033684148808297|0.31381309166736426
# Baseline 2
### Description
The second one is named as model. Spilt each text into single words, count the number of positive words and negative words. If the number of positive words is larger, label as 1=positive, if the number of negative words is larger, label as -1=negative, if equal, label as 0=neutral. 
### Sample Output
```python
print(train_pred[0:10])
```
[1, -1, -1, 0, -1, -1, 0, 0, -1, 0]
### Test Result
Precision, recall,f1-score were used for evaluation. The test result using evaluation metric is
|            |positive    |negative    |neutral     |macroavg 
|:-----------:|-----------------|:------------:|:-----:|:-------:|
|precision   |0.7856927199305582|0.7174790437752251|0.7028251615758082|0.7353323084271972
|recall      |0.5619483358976435|0.4509268292682927|0.8948287217412695|0.6359012956357352
|f1 score        |0.6552467581518318|0.5537982266954229|0.7872895969616789|0.6654448606029778
