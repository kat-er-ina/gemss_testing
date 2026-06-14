#!/usr/bin/env Rscript
# Bridge to the authors' BB-SSL (Nie & Rockova 2020, CRAN 'BBSSL', archived).
# Args: X.csv y.csv out.csv lambda0 lambda1 NSample alpha
# Reads headerless X (n x p) and y (n), runs BB-SSL (method=3, spike-and-slab
# posterior sampling via randomised-objective optimisation), writes the beta
# sample matrix (NSample x p) to out.csv. Regression (Gaussian) model.
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(BBSSL, lib.loc = Sys.getenv("R_LIBS_USER")))
X <- as.matrix(read.csv(args[1], header = FALSE))
y <- as.numeric(read.csv(args[2], header = FALSE)[, 1])
outf <- args[3]
lambda0 <- as.numeric(args[4]); lambda1 <- as.numeric(args[5])
NSample <- as.integer(args[6]); alpha <- as.numeric(args[7])
p <- ncol(X)
res <- BB_SSL(y, X, method = 3, lambda = c(lambda0, lambda1), NSample = NSample,
              a = 1, b = p, maxiter = 500, length.out = 50, burn.in = FALSE,
              discard = TRUE, alpha = alpha, initial.beta = rep(0, p))
write.table(res$beta, outf, sep = ",", row.names = FALSE, col.names = FALSE)
