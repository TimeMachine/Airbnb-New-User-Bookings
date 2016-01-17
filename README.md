## 介紹 ##
### Scripts  ###
由於第一次接觸Machine Learning，一開始對於這題目完全沒有頭緒。
於是參考了許多人的分析和解法:
```
User Data Exploration   by David Gasquez
Airbnb Visualizations   by Terry J
script_0.8655           by Sandro
script                  by FraPochetti
Score predictions using NDCG  by  dietCoke
```
根目錄存在的檔案介紹:

* `baseline.py` : `script_0.8655 by Sandro`

* `main.py` : 我寫的Script

* `NDCG.py` : NDCG量測的函式

* `NDCG_example.py` : 使用`NDCG.py`的範例
### Data exploration ###
關於分析的過程，我放在`Analysis`資料夾。

### Cleaning Data ###
首先關於3個日期欄位的部分:`date_account_created`、`date_first_booking`、`timestamp_first_active`。

我發現將`date_first_booking`加入預測的欄位，會造成NDCG的評分下降，我想主因是缺值太多造成預測不準。

把日期拆成`年、月、週`，會比`年、月、日`還要來的準，我猜因為將日期分的太細會導致預測時，對於日期太敏感而導致誤判。

關於年齡，14歲以下和100歲以上的年齡屬於不正常，於是將不正常、缺值的資料設為中位數。
接著我測量年齡的上界從60到100，發現85歲上下的時候，NDCG的分數是最好的。

除此之外的欄位我並沒有去做調整，全部都放在預測的條件中。

在session data中，我加入考慮`action`次數和各個`device_type`的次數,我認為這對於預測目的地有間接的相關。

### Pick Model ###
我目前所了解的model只有3種:`Random Forest`、`Adaboost`、`Gradient boosting`。
在一般的情況下，`Random Forest`效果會比`Adaboost`還要好，因為時間的關係，所以我不考慮使用`Adaboost`。
於是我有2個model要去測試。

### Optimize Model ###
首先，關於`Gradient boosting`已經有個樣本:`script_0.8655 by Sandro`供我去測試。
我發現我將他的參數中，把樹的數量調高和`learning_rate`調低還可以把NDCG的評分上升，最後上傳test_pred並沒有造成overfit的現象，
於是我知道這個model還沒有到最佳的狀態。最後我的分數提升至`0.86612`

### NDCG Validation ###
關於基本的NDCG驗證已經有人實做出來了，它可以將每筆預測和結果算出NDCG，那我的想法是把每筆資料NDCG作平均來作為我的分數。

我再做一個`Cross Validation Score`的函式把每個`partition`的分數再作平均，而且因為資料分割的特性，可以使用平行計算來節省時間。
關於我所創建的函式做個介紹:
```
def cross_validation_score(x ,labels ,model ,partition):
# x 為總資料的輸入(資料必須為純數字的table)
# labels 為總資料的輸出
# model 為所使用的模型
# partition 將資料切成幾等份來分別算分數
```
詳細的範例放在`NDCG_example.py`中

### Postscript ###
關於`Radom Forest`的model，我還沒有時間去做測試。因為繳交期限和期末考重疊在一起，等我考完試就已經剩不到一個禮拜的時間了。

總結來說，透過其他人的script，我學到了很多關於machine learning的知識，對python和相關的函式庫也變得較熟練。

只考慮train_user的feature是不夠的，分數到0.866就很難再上去了，但是加上考慮session的feature就可以馬上升上0.878。
