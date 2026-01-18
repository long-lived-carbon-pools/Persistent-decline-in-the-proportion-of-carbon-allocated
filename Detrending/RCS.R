library(dplR)
library(tools)

# 创建输出目录
dir.create("D:/Detrending3", showWarnings = FALSE)
dir.create("D:/mistake", showWarnings = FALSE)

# 获取所有 .rwl 文件路径
rwl_files <- list.files("D:/Tree rings/rwl/width", pattern = "\\.rwl$", full.names = TRUE)

# 循环处理每个文件
for (file_path in rwl_files) {
  tryCatch({
    file_name <- basename(file_path)
    csv_name <- file_path_sans_ext(file_name)
    
    # === 读取并跳过前三行 ===
    lines <- readLines(file_path)
    if (length(lines) <= 3) {
      stop("文件数据不足，前三行之后没有内容")
    }
    temp_file <- tempfile(fileext = ".rwl")
    writeLines(lines[-c(1:3)], temp_file)
    rwl_data <- read.rwl(temp_file)
    unlink(temp_file)
    
    # === 去趋势处理 ===
    rwi_negexp <- detrend(rwl_data, method = "ModNegExp")
    rwi_spline <- detrend(rwl_data, method = "Spline", nyrs = 30)
    rwi_agedep <- detrend(rwl_data, method = "AgeDepSpline")
    
    # === 构建年表 ===
    crn_negexp <- chron(rwi_negexp, prefix = "")
    crn_spline <- chron(rwi_spline, prefix = "")
    crn_agedep <- chron(rwi_agedep, prefix = "")
    
    # === 检查年表是否构建成功 ===
    if (any(is.na(crn_negexp$std)) || any(is.na(crn_spline$std)) || any(is.na(crn_agedep$std))) {
      stop("年表构建失败")
    }
    
    # === 年份 ===
    years <- as.numeric(rownames(crn_agedep))
    
    # === 年表转为数据框 ===
    df_negexp <- data.frame(year = years, ModNegExp = crn_negexp$std, samp.depth = crn_negexp$samp.depth)
    df_spline <- data.frame(year = years, Spline = crn_spline$std)
    df_agedep <- data.frame(year = years, AgeDepSpline = crn_agedep$std)
    
    # === RCS（区域曲线标准化）===
    rw <- as.matrix(rwl_data)
    ages <- apply(rw, 2, function(x){ idx <- which(!is.na(x)); a <- rep(NA_real_, length(x)); if(length(idx)>0) a[idx] <- seq_along(idx); a })
    max_age <- suppressWarnings(max(ages, na.rm = TRUE))
    rc <- sapply(seq_len(max_age), function(a) mean(rw[ages == a], na.rm = TRUE))
    rwi_rcs <- rw
    for(j in seq_len(ncol(rw))){
      aj <- ages[,j]
      ok <- !is.na(rw[,j]) & !is.na(aj)
      rwi_rcs[ok,j] <- rw[ok,j] / rc[aj[ok]]
      rwi_rcs[!ok,j] <- NA_real_
    }
    rwi_rcs <- as.data.frame(rwi_rcs); rownames(rwi_rcs) <- rownames(rwl_data); colnames(rwi_rcs) <- colnames(rwl_data)
    crn_rcs <- chron(rwi_rcs, prefix = "")
    df_rcs <- data.frame(year = as.numeric(rownames(crn_rcs)), RCS = crn_rcs$std)
    
    # === 合并三种方法 ===
    merged_crn <- Reduce(function(x, y) merge(x, y, by = "year", all = TRUE),
                         list(df_negexp, df_spline, df_agedep))
    merged_crn <- merge(merged_crn, df_rcs, by = "year", all = TRUE)
    
    # === 计算逐年 EPS（手动）===
    eps_vec <- rep(NA, nrow(rwi_agedep))
    rwi_years <- as.numeric(rownames(rwi_agedep))
    
    for (i in seq_along(rwi_years)) {
      values <- rwi_agedep[i, ]
      valid_samples <- which(!is.na(values))
      n <- length(valid_samples)
      if (n >= 2) {
        mat <- rwi_agedep[, valid_samples]
        cor_mat <- suppressWarnings(cor(mat, use = "pairwise.complete.obs"))
        r_vals <- cor_mat[lower.tri(cor_mat)]
        rbar <- mean(r_vals, na.rm = TRUE)
        eps_vec[i] <- (n * rbar) / (n * rbar + (1 - rbar))
      }
    }
    
    eps_df <- data.frame(year = rwi_years, EPS = eps_vec)
    
    # === 合并 EPS ===
    merged_crn <- merge(merged_crn, eps_df, by = "year", all = TRUE)
    
    # === 保存输出 ===
    write.csv(merged_crn, file = paste0("D:/Detrending3/", csv_name, ".csv"), row.names = FALSE)
    
    message("✅ 成功处理：", file_name)
    
  }, error = function(e) {
    file.copy(file_path, paste0("D:/mistake/", basename(file_path)), overwrite = TRUE)
    message("❌ 失败：", basename(file_path), "，错误信息：", e$message)
  })
}
