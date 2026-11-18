# Plan sec. 39: fit SuSiE (Wang, Sarkar, Carbonetto & Stephens 2020) with the reference implementation susieR,
# at its defaults (standardize, intercept, residual and prior variances estimated, coverage 0.95, min_abs_corr 0.5).
# Called by susie_compare.py.  Usage: Rscript susie_fit.R <job dir> [<job dir> ...]
# Each job dir holds dims.txt ("n p L"), X.bin (n x p float64, column-major), y.bin (n float64).
# Writes alpha.bin (L x p float64, column-major), pip.bin (p float64), cs.txt (one line per credible set:
# "<effect, 0-based> <min.abs.corr> <members, 0-based, comma-separated>") and info.txt (key value lines).
suppressMessages(library(susieR))
args <- commandArgs(trailingOnly = TRUE)
for (d in args) {
  dims <- scan(file.path(d, "dims.txt"), quiet = TRUE)
  n <- dims[1]; p <- dims[2]; L <- dims[3]
  X <- matrix(readBin(file.path(d, "X.bin"), "double", n * p, size = 8, endian = "little"), n, p)
  y <- readBin(file.path(d, "y.bin"), "double", n, size = 8, endian = "little")
  t0 <- proc.time()[["elapsed"]]
  fit <- susie(X, y, L = L)
  secs <- proc.time()[["elapsed"]] - t0
  writeBin(as.vector(fit$alpha), file.path(d, "alpha.bin"), size = 8, endian = "little")
  writeBin(as.vector(fit$pip), file.path(d, "pip.bin"), size = 8, endian = "little")
  cs <- fit$sets
  lines <- character(0)
  if (!is.null(cs$cs) && length(cs$cs) > 0) {
    for (i in seq_along(cs$cs)) {
      lines <- c(lines, paste(cs$cs_index[i] - 1, sprintf("%.6f", cs$purity[i, "min.abs.corr"]),
                              paste(cs$cs[[i]] - 1, collapse = ",")))
    }
  }
  writeLines(lines, file.path(d, "cs.txt"))
  writeLines(c(sprintf("sigma2 %.10g", fit$sigma2), sprintf("elbo %.10g", tail(fit$elbo, 1)),
               sprintf("niter %d", as.integer(fit$niter)), sprintf("converged %d", as.integer(isTRUE(fit$converged))),
               sprintf("seconds %.4f", secs), paste("V", paste(sprintf("%.6g", fit$V), collapse = ",")),
               paste("version", as.character(packageVersion("susieR")))), file.path(d, "info.txt"))
}
