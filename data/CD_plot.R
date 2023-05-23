library(dplyr)
library(scmamp)

df <- read.csv(
  "./data.csv", 
  sep="\t", dec=','
)
rownames(df) <- df$X
df <- df[, -1]
colnames(df) <- c(
  "e-PoI", "EHVI", "e-PoHVI-smoothing", "e-PoHVI-scaling"
)

pdf("ZDT.pdf", width = 9, height = 3)
plotCD(
  filter(df, startsWith(rownames(df), "ZDT")), 
  alpha = 0.01, cex = 1
)
dev.off()

pdf("WOSGZ.pdf", width = 9, height = 2)
plotCD(
  filter(df, startsWith(rownames(df), "WOSGZ")), 
  alpha = 0.01, cex = 1
)
dev.off()

pdf("RE.pdf", width = 9, height = 3)
plotCD(
  filter(df, startsWith(rownames(df), "RE")), 
  alpha = 0.01, cex = 1
)
dev.off()

pdf("All.pdf", width = 9, height = 3)
plotCD(
  df, 
  alpha = 0.01, cex = 1
)
dev.off()