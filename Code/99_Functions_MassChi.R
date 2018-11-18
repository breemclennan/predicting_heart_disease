# REF: https://stackoverflow.com/questions/7382039/chi-square-analysis-using-for-loop-in-r
# It creates a matrix where all nominal variables are tested against each other.
# It can also save the results as excel file. It displays all the pvalues that are smaller than 5%.
# delFirst parameter can delete the first n columns. So if you have an count index or something you dont want to test.
# Can also specify a path for XL output

funMassChi <- function (x,delFirst=0,xlsxpath=FALSE) {
  options(scipen = 999)
  
  start <- (delFirst+1)
  ds <- x[,start:ncol(x)]
  
  cATeND <- ncol(ds)
  catID  <- 1:cATeND
  
  resMat <- ds[1:cATeND,1:(cATeND-1)]
  resMat[,] <- NA
  
  for(nCc in 1:(length(catID)-1)){
    for(nDc in (nCc+1):length(catID)){
      tryCatch({
        chiRes <- chisq.test(ds[,catID[nCc]],ds[,catID[nDc]])
        resMat[nDc,nCc]<- chiRes[[3]]
      }, error=function(e){cat(paste("ERROR :","at",nCc,nDc, sep=" "),conditionMessage(e), "\n")})
    }
  }
  resMat[resMat > 0.05] <- "" 
  Ergebnis <- cbind(CatNames=names(ds),resMat)
  Ergebnis <<- Ergebnis[-1,] 
  
  if (!(xlsxpath==FALSE)) {
    write.xlsx(x = Ergebnis, file = paste(xlsxpath,"ALLChi-",Sys.Date(),".xlsx",sep=""),
               sheetName = "Tabelle1", row.names = FALSE)
  }
}
