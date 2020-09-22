library(cosinor2)
library(openxlsx)
library(ggplot2)
library(magrittr)
library(stringr)
library(purrr)

options(stringsAsFactors=FALSE)

file_name = "test_data/data.xlsx"


i <- 1
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
df.time <- as.numeric(mydf[,1])
df.data <- mydf[,-1]
mydf <- data.frame(df.time, df.data)
colnames(mydf) <- c('time', 'Y')
test1 <- mydf
test1$X <- 0

i <- 2
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
df.time <- as.numeric(mydf[,1])
df.data <- mydf[,-1]
mydf <- data.frame(df.time, df.data)
colnames(mydf) <- c('time', 'Y')
test2 <- mydf
test2$X <- 1

test12 = rbind(test1, test2)
fit12 <- cosinor.lm(Y ~ time(time) + X + amp.acro(X), data = test12, period = 24)

i <- 3
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
df.time <- as.numeric(mydf[,1])
df.data <- mydf[,-1]
mydf <- data.frame(df.time, df.data)
colnames(mydf) <- c('time', 'Y')
test3 <- mydf
test3$X <- 0

i <- 4
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
df.time <- as.numeric(mydf[,1])
df.data <- mydf[,-1]
mydf <- data.frame(df.time, df.data)
colnames(mydf) <- c('time', 'Y')
test4 <- mydf
test4$X <- 1

test34 = rbind(test3, test4)
fit34 <- cosinor.lm(Y ~ time(time) + X + amp.acro(X), data = test34, period = 24)


t12_amp <- test_cosinor(fit12, "X", param = "amp")
t12_acr <- test_cosinor(fit12, "X", param = "acr")
t34_amp <- test_cosinor(fit34, "X", param = "amp")
t34_acr <- test_cosinor(fit34, "X", param = "acr")


s12 <- summary(fit12)$transformed.table
s34 <- summary(fit34)$transformed.table

estimates12 <- s12$estimate
ps12 <- s12$p.value
estimates34 <- s34$estimate
ps34 <- s34$p.value

test <- c('test1 vs. test2', 'test3 vs. test4')

amplitude1 <- c(estimates12[3], estimates34[3])
p_amplitude1 <- c(ps12[3], ps34[3])
amplitude2 <- c(estimates12[4], estimates34[4])
p_amplitude2 <- c(ps12[4], ps34[4])
d_amplitude <- amplitude2-amplitude1
p_d_amplitude <- c(t12_amp$global.test$p.value, t34_amp$global.test$p.value)

acrophase1 <- c(estimates12[5], estimates34[5])
p_acrophase1 <- c(ps12[5], ps34[5])
acrophase2 <- c(estimates12[6], estimates34[6])
p_acrophase2 <- c(ps12[6], ps34[6])
d_acrophase <- acrophase2-acrophase1
p_d_acrophase <- c(t12_acr$global.test$p.value, t34_acr$global.test$p.value)

res <- data.frame(test, amplitude1, p_amplitude1, amplitude2, p_amplitude2, d_amplitude, p_d_amplitude, acrophase1, p_acrophase1, acrophase2, p_acrophase2, d_acrophase, p_d_acrophase)
                  
                  
write.csv(res, "test_data/supp_table_6.csv", row.names = FALSE)