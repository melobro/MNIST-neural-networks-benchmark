data1 <- read.table("bchmk_llb1_t.no.1.txt")
data2 <- read.table("bchmk_llb1_t.no.2.txt")
data4 <- read.table("bchmk_llb1_t.no.4.txt")
data8 <- read.table("bchmk_llb1_t.no.8.txt")
samples1 <- data1$V1
time1 <- data1$V2
samples2 <- data2$V1
time2 <- data2$V2
samples4 <- data4$V1
time4 <- data4$V2
samples8 <- data8$V1
time8 <- data8$V2
pdf("test_llb1_bchmk.pdf")
 plot(time8, samples8, type = "l", xlab = "Time in sec", ylab = "No. of Samples", main = "test_llb1_bchmk", col= "red")
lines(time4, samples4, col="blue")
lines(time2, samples2, col="green")
lines(time1, samples1, col="black")
legend("topleft", legend=c("1-Threads", "2-Threads", "4-Threads", "8-Threads"), col=c("black", "green", "blue", "red"), lty=1)
dev.off()
quit()
