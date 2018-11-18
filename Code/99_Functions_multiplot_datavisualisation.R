
#REF:https://align-alytics.com/drawing-a-grid-of-plots-in-r-regression-lines-loess-curves-and-more/
#Mosaic plots include the p-value of a chi-square test of independence: p < 0.05 indicates 
#that there is a significant relationship between the two variables under consideration.
#The number of plot cells with a count under five is shown; if this is greater than zero, 
#the chi-square test may be invalid. Here's an example using a continuous target variable: 

#Usage with regression, continuous values
#mtcars2 <- mtcars
#mtcars2$cyl  <- as.factor(mtcars2$cyl)
#mtcars2$vs   <- as.factor(mtcars2$vs)
#mtcars2$am   <- as.factor(mtcars2$am)
#mtcars2$gear <- as.factor(mtcars2$gear)
#mtcars2$carb <- as.factor(mtcars2$carb)
#multiplot(mtcars2, 'disp', c(2, 5))

# Factor/Category Usage:
#This example has a categorical target variable:
# multiplot(mtcars2, 'gear', c(2, 5))
#======#

multiplot <- function(df_data, y_column, mfrow=NULL){
  #
  # Plots the data in column y_column of df_data against every other column in df_data, a dataframe.
  # By default the plots are drawn next to each other (i.e. in a row). Use mfrow to overide this. E.g. mfrow=c(2, 3). 
  #
  
  # Set the layout
  if (is.null(mfrow)) mfrow <- c(1, ncol(df_data) - 1)
  op <- par(mfrow=mfrow, mar=c(5.1, 4.1, 1.1, 1.1), mgp = c(2.2, 1, 0))
  on.exit(par(op))
  
  for (icol in which(names(df_data) != y_column)){
    x_column <- names(df_data)[icol]
    y_x_formula <- as.formula(paste(y_column, "~", x_column))
    x_y_formula <- as.formula(paste(x_column, "~", y_column))
    x <- df_data[[x_column]]
    y <- df_data[[y_column]]
    subtitle <- ""
    
    if (is.factor(x)){
      if (is.factor(y)){
        # Mosaic plot.
        tbl <- table(x, y)
        chi_square_test_p <- chisq.test(tbl)$p.value
        problem_cell_count <- sum(tbl < 5)
        subtitle <- paste("Chi-Sq. Test P:", round(chi_square_test_p, 5)," (< 5 in ", problem_cell_count, " cells.)")
        plot(y_x_formula, data=df_data)
      } else {
        # Vertical boxplot.
        fit <- aov(y_x_formula, data=df_data)
        f_test_p <- summary(fit)[[1]][["Pr(>F)"]][[1]]
        subtitle <- paste("F-Test P:", round(f_test_p, 5))
        boxplot(y_x_formula, data=df_data, horizontal=FALSE, varwidth=TRUE)
        means <- tapply(y, x, function(z){mean(z, na.rm=TRUE)})
        points(x=means, col="red", pch=18)
      }
    } else {
      if (is.factor(y)){
        # Horizontal boxplot.
        fit <- aov(x_y_formula, data=df_data)
        f_test_p <- summary(fit)[[1]][["Pr(>F)"]][[1]]
        subtitle <- paste("F-Test P:", round(f_test_p, 5))
        boxplot(x_y_formula, data=df_data, horizontal=TRUE, varwidth=TRUE)
        means <- tapply(x, y, function(z){mean(z, na.rm=TRUE)})
        points(x=means, y=1:length(levels(y)), col="red", pch=18)
      } else {
        # Scatterplot with straight-line regression and lowess line.
        adj_r_squared <- summary(lm(y_x_formula, df_data))$adj.r.squared
        subtitle <- paste("Adj. R Squared:", round(adj_r_squared, 5))
        plot(y_x_formula, data=df_data, pch=19, col=rgb(0, 0, 0, 0.2))
        abline(lm(y_x_formula, data=df_data), col="red", lwd=2)
        lines(lowess(x=x, y=y), col="blue", lwd=2) 
      }
    }
    title(sub=subtitle, xlab=x_column, ylab=y_column)
  }
}