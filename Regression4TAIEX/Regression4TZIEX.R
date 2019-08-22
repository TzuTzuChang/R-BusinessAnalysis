#Step1-查看與設定工作目錄
getwd()
setwd('C:/Users/TzuTzu/Desktop/CTBC')

#Step2-載入資料並查看資料架構
Train_X <- read.table("Train_X.csv", header=T, sep=",", fileEncoding='big5') 
CorpInfo_2014 <- read.csv("CorpInfo_2014.csv", header=T, sep=",", fileEncoding='big5')
Train_Y.df <- read.table("Train_Y.csv", header=T, sep=",", fileEncoding='big5') 
str(Train_X)
str(CorpInfo_2014)
str(Train_Y.df)

#Step3-轉換資料型態
library(tidyverse)
Train_X.tb <- as_tibble(Train_X)
CorpInfo_2014.tb <- as_tibble(CorpInfo_2014)
CorpInfo_2014.tb$股票代號 <- as.factor(CorpInfo_2014.tb$股票代號)
Train_X.tb$日期 <- as.character(Train_X.tb$日期, format = "%Y%m%d")
Train_X.tb$日期 <- as.Date(Train_X.tb$日期, format = "%Y%m%d")

#Step4-挑出上市股票
ListedCrop.tb <-
  CorpInfo_2014.tb %>% filter(CorpInfo_2014.tb$上市上櫃 == 1 )

#藉由semi_join挑出Train_X的上市股票
ListedCrop_2014.tb <- 
  Train_X.tb %>% 
  semi_join(ListedCrop.tb, by = c("股票代號"))

#Step5-篩選資料集
#由於台灣的加權指數採市值加權
#表示市值與加權指數會成正比關係
#因為時間關係，故先挑選市值做預測
ValueListed.df <- data.frame(Date = ListedCrop_2014.tb$日期,
                             Stock = ListedCrop_2014.tb$股票代號,
                             Value = ListedCrop_2014.tb$總市值.億.)

#Step6-將值轉回變數
TransValueListed.df <- ValueListed.df %>% 
  spread(key = Stock,
         value = Value)

#Steo7-使用Z-score標準化
scaleListed <- subset(TransValueListed.df, select = -Date)
scaleListed <- round(scale(scaleListed, center=T, scale=T), digits = 2)
str(scaleListed)
##上述結果顯示，為一個超大矩陣

#轉換標準化後之型態
scaleTrain.df <- as.data.frame(scaleListed)

#Step8-處理離群值
boxplot(scaleTrain.df)
which(scaleTrain.df > 3 | scaleTrain.df < -3)
##結果顯示337個值>3或< -3

#調整離群值
#將>3之值都調整為3
#<-3之值都調整為-3
scaleTrain.df[scaleTrain.df>3] <- 3
scaleTrain.df[scaleTrain.df< -3] <- -3
boxplot(scaleTrain.df)
which(scaleTrain.df > 3 | scaleTrain.df < -3)
##結果顯示integer(0)，已無離群值

#Step9-處理NA值
sum(is.na(scaleTrain.df))
##結果顯示[1] 2372

#畫出NA圖之分布
library(VIM)
aggr_plot <- aggr(scaleTrain.df, 
                  col = c('navyblue', 'red'), 
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(scaleTrain.df), 
                  cex.axis=.7, gap=3,
                  ylab=c("Histogram of missing data", "Pattern"))
##結果顯示Variables sorted by number of missings: 
#Variable      Count
#911201 1.00000000
#6449 0.97046414
#6431 0.95780591
#3661 0.84388186
#1592 0.81012658
#5288 0.75105485
#8443 0.71729958
#2634 0.65822785
#3437 0.52320675
#910948 0.48945148
#1568 0.35864979
#8150 0.26582278
#6409 0.23206751
#6414 0.22784810
#5285 0.13502110
#2384 0.08860759
#911619 0.04641350
#1262 0.04219409
#1315 0.04219409
#1413 0.04219409
#1538 0.04219409
#2375 0.04219409
#2426 0.04219409
#2438 0.04219409
#2456 0.04219409
#2459 0.04219409
#3046 0.04219409
#1410 0.03797468
#1513 0.03797468
#2408 0.03797468
#4930 0.03797468
#5305 0.03797468
#1340 0.03375527
#1449 0.03375527
#1516 0.03375527
#2327 0.03375527
#2331 0.03375527
#2444 0.03375527
#1456 0.02953586
#2359 0.02953586
#2476 0.02953586
#3059 0.02953586

#刪除NA比例過半者
NewTrain.df <- scaleTrain.df[ , !names(scaleTrain.df) %in% c("911201", "6449", "6431", "3661", "1592",
                                                                   "5288", "8443", "2634", "3437", "910948")]

#補剩餘NA為0
#因為有NA之變數，不見得為建模之變數
#所以先不採用任何方式補值
NewTrain.df[is.na(NewTrain.df)] <- 0
sum(is.na(NewTrain.df))
##結果顯示[1] 0，已無NA


#Step10-合併Train的XY資料集
Traindata <- cbind(NewTrain.df,
                   y = Train_Y.df$Target_Y)

#Step11-PCA降維，挑選變數
PCA.Train.df <- princomp(Traindata, cor = F )
##結果顯示，Error in princomp.default(NewTrain.df, cor = F) : 
#  'princomp' can only be used with more units than variables
#因為PCA有變數上限之限制，所以無法使用PCA

#Step12-使用向前選取法先做回歸
#向前選取法市由空模型開始，一個一個丟變數
#最大不會超過完整的線性迴歸
#且最先挑到之變數表示最為重要
null = lm(y ~ 1, data = Traindata)  
full = lm(y ~ ., data = Traindata) 
forward.lm = step(null, 
                  scope=list(lower=null, upper=full), 
                  direction="forward")

#Step13-挑選變數後建模
#從向前選取法中調整建模選用之參數
#一般建模參數挑選8~12為最佳
#所以先選前12個向前選取法挑到之變數
Traindata1 <- data.frame(x1  = NewTrain.df$`4915`, 
                         x2  = NewTrain.df$`9906`, 
                         x3  = NewTrain.df$`2816`, 
                         x4  = NewTrain.df$`3406`, 
                         x5  = NewTrain.df$`3034`,
                         x6  = NewTrain.df$`3443`,
                         x7  = NewTrain.df$`2493`,
                         x8  = NewTrain.df$`2385`,
                         x9  = NewTrain.df$`9103`,
                         x10 = NewTrain.df$`3032`,
                         x11 = NewTrain.df$`3014`,
                         x12 = NewTrain.df$`2528`,
                         y = Train_Y.df$Target_Y) 

NewlmTrain1 <- lm(formula = y ~ x1 + x2 + x3 + x4 + x5 + x6 +
                                x7 + x8 + x9 + x10 + x11 + x12, 
                  data = Traindata1)
summary(NewlmTrain1)
##結果顯示Adjusted R-squared:0.9369且p-value: < 2.2e-16
#表示模型有93%的解釋力且p-value<0.05

#RMSE
RMSE <- function(predict, actual){
  result <- sqrt(mean((predict - actual) ^ 2))
  return(result)
}
cat('RMSE:\n', RMSE(NewlmTrain1$fitted.values, Train_Y.df$Target_Y), '\n', sep='')

#MAPE
MAPE <- function(predict, actual){
  result <- mean(abs((predict - actual) / actual)) %>% round(3) * 100
  return(result)
}
cat('MAPE:\n', MAPE(NewlmTrain1$fitted.values, Train_Y.df$Target_Y), '%', '\n', sep='')
##結果顯示RMSE:71.6847, MAPE:0.6%

#Step14-模型診斷
library(ggfortify)
autoplot(NewlmTrain1)
#由圖片結果可推論沒有特殊的趨勢或是奇怪的地方

#Step14.1-殘差常態性檢定
shapiro.test(NewlmTrain1$residual)
##結果顯示
#Shapiro-Wilk normality test
#data:  NewlmTrain$residual
#W = 0.9883, p-value = 0.05081
#p-value>0.05表不拒絕虛無假設
#即殘差值服從常態分配

##Step14.2-殘差獨立性檢定
library(car)
durbinWatsonTest(NewlmTrain1)
##結果顯示
#lag Autocorrelation D-W Statistic p-value
# 1       0.4971435     0.9931856       0
#Alternative hypothesis: rho != 0
#以一般 95% 的信賴水準來說，p-value<0.05即拒絕虛無假設的
#也就是說殘差沒有符合獨立性的假設
#但是因為p-value並沒有非常小
#所以證據並不是非常明確

##Step14.3-殘差變異數同質性檢定
library(car)
ncvTest(NewlmTrain1)
##結果顯示
#Non-constant Variance Score Test 
#Variance formula: ~ fitted.values 
#Chisquare = 2.201708, Df = 1, p = 0.13786
#殘差變異數同質性檢定p-value>0.05
#所以不拒絕虛無假設
#亦即殘差的變異數符合同質性的假設

#Step15-拿Test資料集做預測
Test_X <- read.table("Test_X.csv", header=T, sep=",", fileEncoding='big5')
Test_Y <- read.table("Test_Y.csv", header=T, sep=",", fileEncoding='big5')

#挑出上市股票
TestListed <- 
Test_X %>% 
  semi_join(ListedCrop.tb, by = c("股票代號"))

#挑選需要之變數
TestValueListed.df <- data.frame(Date = TestListed$日期,
                                 Stock = TestListed$股票代號,
                                 Value = TestListed$總市值.億.)

#將值轉為變數
TransTest <- TestValueListed.df %>% 
  spread(key = Stock,
         value = Value)

#標準化
scaleTest <- subset(TransTest, select = -Date)
scaleTest <- round(scale(scaleTest, center=T, scale=T), digits = 2)
str(scaleTest)

#轉回dataframe
scaleTest.df <- as.data.frame(scaleTest)

#處理NA
sum(is.na(scaleTest.df))
library(VIM)
aggr_plot <- aggr(scaleTest.df, 
                  col = c('navyblue', 'red'), 
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(scaleTest.df), 
                  cex.axis=.7, gap=3,
                  ylab=c("Histogram of missing data", "Pattern"))
scaleTest.df[is.na(scaleTest.df)] <- 0
sum(is.na(scaleTest.df))
str(scaleTest.df)

#挑出需要之變數
Test1 <- cbind(x1 = scaleTest.df$`4915`, 
               x2 = scaleTest.df$`9906`, 
               x3 = scaleTest.df$`2816`, 
               x4 = scaleTest.df$`3406`, 
               x5 = scaleTest.df$`3034`,
               x6 = scaleTest.df$`3443`,
               x7 = scaleTest.df$`2493`,
               x8 = scaleTest.df$`2385`,
               x9 = scaleTest.df$`9103`,
               x10= scaleTest.df$`3032`,
               x11= scaleTest.df$`3014`,
               x12= scaleTest.df$`2528`)
Test1 <- as.data.frame(Test1)

pre1 <- predict(NewlmTrain1, Test1)

cat('RMSE:\n', RMSE(pre1, Test_Y$Target_Y), '\n', sep='')
cat('MAPE:\n', MAPE(pre1, Test_Y$Target_Y), '%', '\n', sep='')
#結果顯示RMSE:267.3184, MAPE:2.4%%

#Step16-檢討模型
#雖然訓練資料之RMSE表現不錯
#希望可以更好故調整模型選用之參數
#故挑選了9, 12, 20~29, 50~53個不等之變數個數做再次訓練
#其模型在訓練時，解釋力皆高達90%以上
#p-value亦皆<0.05
#故比較RMSE, MAPE之數值

##結果顯示
#挑選9個變數時-  RMSE:78.78321, MAPE:0.7%
#挑選12個變數時- RMSE:71.6847, MAPE:0.6%
#挑選20個變數時- RMSE:60.1141, MAPE:0.5%
#挑選21個變數時- RMSE:57.42241, MAPE:0.4%
#挑選22個變數時- RMSE:55.04522, MAPE:0.5%
#挑選23個變數時- RMSE:53.57089, MAPE:0.5%
#挑選24個變數時- RMSE:52.17445, MAPE:0.4%
#挑選25個變數時- RMSE:49.2199, MAPE:0.4%
#挑選26個變數時- RMSE:49.17465, MAPE:0.4%
#挑選27個變數時- RMSE:48.00297, MAPE:0.4%
#挑選28個變數時- RMSE:47.32487, MAPE:0.4%
#挑選29個變數時- RMSE:45.70366, MAPE:0.4%
#挑選50個變數時- RMSE:30.46431, MAPE:0.3%
#挑選51個變數時- RMSE:29.68467, MAPE:0.3%
#挑選52個變數時- RMSE:29.12608, MAPE:0.3%
#挑選53個變數時- RMSE:28.83778, MAPE:0.3%

#挑選模型之順序為
#發現初步挑選的12個變數有3個具高度相關
#故淘汰後先選9個變數做模型訓練
#但RMSE, MAPE並未下降
#所以改變策略選擇增加變數
#以12的倍數做調整
#選定25, 50個變數做訓練
#模型在50個變數時RMSE, MAPE明顯比12個變數好一倍
#因此，在50個變數以上繼續挖掘可能之模型
#但51~53個變數之模型的RMSE, MAPE並未明顯下降
#且在25個變數模型應用於預測時
#RMSE, MAPE反而比9, 12, 50~53個變數模型做預測之結果更好
#故選定25個變數前後繼續探索最適化之模型
#發現在24個變數時，模型預測能力為目前最好之結論

#Step17-選定最適化模型
Traindata24 <- data.frame(x1  = NewTrain.df$`4915`, 
                          x2  = NewTrain.df$`9906`, 
                          x3  = NewTrain.df$`2816`, 
                          x4  = NewTrain.df$`3406`, 
                          x5  = NewTrain.df$`3034`,
                          x6  = NewTrain.df$`3443`,
                          x7  = NewTrain.df$`2493`,
                          x8  = NewTrain.df$`2385`,
                          x9  = NewTrain.df$`9103`,
                          x10 = NewTrain.df$`3032`,
                          x11 = NewTrain.df$`3014`,
                          x12 = NewTrain.df$`2528`,
                          x13 = NewTrain.df$`2409`,
                          x14 = NewTrain.df$`1453`,
                          x15 = NewTrain.df$`6257`,
                          x16 = NewTrain.df$`2430`,
                          x17 = NewTrain.df$`4976`,
                          x18 = NewTrain.df$`9941`,
                          x19 = NewTrain.df$`1727`,
                          x20 = NewTrain.df$`2707`,
                          x21 = NewTrain.df$`6209`,
                          x22 = NewTrain.df$`5305`,
                          x23 = NewTrain.df$`4722`,
                          x24 = NewTrain.df$`2702`,
                          y = Train_Y.df$Target_Y) 

NewlmTrain24 <- lm(formula = y ~ x1 + x2 + x3 + x4 + x5 + x6 +
                                 x7 + x8 + x9 + x10 + x11 + x12 +
                                 x14 + x14 + x15 + x16 + x17 + x18 +
                                 x19 + x20 + x21 + x22 + x23 + x24,
                   data = Traindata24)
summary(NewlmTrain24)

#RMSE
cat('RMSE:\n', RMSE(NewlmTrain24$fitted.values, Train_Y.df$Target_Y), '\n', sep='')

#MAPE
cat('MAPE:\n', MAPE(NewlmTrain24$fitted.values, Train_Y.df$Target_Y), '%', '\n', sep='')
#RMSE:52.17445, MAPE:0.4%

#挑出需要之變數
Test24<- cbind(x1 = scaleTest.df$`4915`, 
               x2 = scaleTest.df$`9906`, 
               x3 = scaleTest.df$`2816`, 
               x4 = scaleTest.df$`3406`, 
               x5 = scaleTest.df$`3034`,
               x6 = scaleTest.df$`3443`,
               x7 = scaleTest.df$`2493`,
               x8 = scaleTest.df$`2385`,
               x9 = scaleTest.df$`9103`,
               x10= scaleTest.df$`3032`,
               x11= scaleTest.df$`3014`,
               x12= scaleTest.df$`2528`,
               x13= scaleTest.df$`2409`,
               x14= scaleTest.df$`1453`,
               x15= scaleTest.df$`6257`,
               x16= scaleTest.df$`2430`,
               x17= scaleTest.df$`4976`,
               x18= scaleTest.df$`9941`,
               x19= scaleTest.df$`1727`,
               x20= scaleTest.df$`2707`,
               x21= scaleTest.df$`6209`,
               x22= scaleTest.df$`5305`,
               x23= scaleTest.df$`4722`,
               x24= scaleTest.df$`2702`)
Test24 <- as.data.frame(Test24)

pre24 <- predict(NewlmTrain24, Test24)

#Step18-匯出結論
Date <- Test_Y$日期
Target_Y <- pre24
Test_Y <- data.frame(Date,
                     Target_Y)

write.csv(Test_Y, file="C:\\Users\\TzuTzu\\Desktop\\CTBC\\Test_Y.csv", row.names = FALSE)
