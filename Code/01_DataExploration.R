# Data Exploration.R

# 02_Refined_MLAlgorithmEvaluation_XGBoost_pure_FEATURE REMOVAL.R

# Objective: improve on previous XGBoost pure, performance outcome by removing features identified of lesser relevance.

options(scipen = 999) 
seed <- 1234
set.seed(seed)

library(tidyverse) # data manipulation
library(mlr)       # ML package (also some data manipulation)
library(mlrMBO)
library(DiceKriging)
library(knitr)     # just using this for kable() to make pretty tables
library(xgboost)
library(dplyr)
library(purrr)
library(rprojroot)
library(data.table)
library(doParallel)
library(Metrics) # For calculating logloss evaluation metrics
library(caret)   # for additional data prep functions
library(skimr)
library(DescTools)
library(RDCOMClient)
library(gridExtra) #viewing multiple plots together
library(lattice)
library(knitr) # for dynamic reporting
library(DataExplorer)
#TODO: Add all the other libraries here.


`%ni%` <- Negate(`%in%`)

# Setup file and folder paths for project
# Define a function that computes file paths relative to where root R project folder is located
F <- is_rstudio_project$make_fix_file() 


## DataExplorer: Customize report configuration
config <- list(
  "introduce" = list(),
  "plot_str" = list(
    "type" = "diagonal",
    "fontSize" = 35,
    "width" = 1000,
    "margin" = list("left" = 350, "right" = 250)
  ),
  "plot_missing" = list(),
  "plot_histogram" = list(),
  "plot_qq" = list(sampled_rows = 1000L),
  "plot_bar" = list(),
  "plot_correlation" = list("cor_args" = list("use" = "pairwise.complete.obs")),
  "plot_prcomp" = list(),
  "plot_boxplot" = list(),
  "plot_scatterplot" = list(sampled_rows = 1000L)
)


# The 'xgboost' library must be installed - doesn't need to be loaded
#train_orig <- read_csv("xgboost_train.csv")
#test_orig <- read_csv("xgboost_test.csv")

# Read in CSV
# TRAINING SETS #
raw.train_labels <- read.csv(F("/Data/train_labels.csv"), stringsAsFactors = FALSE)
raw.train_values <- read.csv(F("/Data/train_values.csv"), stringsAsFactors = FALSE)

# TESTING SETS
raw.test_values <- read.csv(F("/Data/test_values.csv"), stringsAsFactors = FALSE)


# Attach training values to training labels by primary key "patient_id"
wrk.train <- list(raw.train_labels, raw.train_values) %>%
  reduce(left_join, by = c("patient_id" = "patient_id")) %>%
  mutate(CATDataSetOrigin = "Training")

# setup the test dataset ready to append by row, to the training set.
wrk.test <- raw.test_values %>%
  mutate(CATDataSetOrigin =  "Testing") %>%
  mutate(heart_disease_present = NA) # create the column for target but keep NULL.


# Append all above raw datasets together: 
# WARNING: columns are different between datasets because test doesn't have a target variable.
# Using rbind + FILL option
wrk.combined <- rbind(setDT(wrk.train), setDT(wrk.test), fill = TRUE)

# 1. Dimensions of the dataset
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the dim function.
dim(wrk.combined)

# 2.  Types of attributes
# It is a good idea to get an idea of the types of the attributes. 
# They could be doubles, integers, strings, factors and other types.
# Knowing the types is important as it will give you an idea
# of how to better summarize the data you have and the types of transforms you might need to use to prepare the data before you model it.
# list types for each attribute
sapply(wrk.combined, class)

# 3. Peek at the data itself
head(wrk.combined)

mlr::summarizeColumns(wrk.combined) %>%
  kable(digits = 2)


# We have some character and numeric fields which can be recoded as factors
# Potential feature selection for furture iterations:
# thal, chest_pain_type asymptomatic, ST depression significant

wrk.combined_datatypecleaned <- wrk.combined %>%
  mutate(NUMthal = case_when(thal == "normal" ~ 3,
                             thal == "reversible_defect" ~ 6,
                             thal == "fixed_defect" ~ 7 )) %>%
  mutate(CATsex = case_when(sex == 0 ~ "female",
                            sex == 1 ~ "male")) %>%    
  mutate(CATchest_pain_type = case_when(chest_pain_type == 1 ~ "typical_angina",
                                        chest_pain_type == 2 ~ "atypical_angina",
                                        chest_pain_type == 3 ~ "non-anginal_pain",
                                        chest_pain_type == 4 ~ "asymptomatic")) %>%
  mutate(CATslope_of_peak_exercise_st_segment = case_when(slope_of_peak_exercise_st_segment == 1 ~ "upsloping",
                                                          slope_of_peak_exercise_st_segment == 2 ~ "flat",
                                                          slope_of_peak_exercise_st_segment == 3 ~ "downsloping")) %>%
  mutate(CATfasting_blood_sugar_gt_120_mg_per_dl = case_when(fasting_blood_sugar_gt_120_mg_per_dl == 0 ~ "no",
                                                             fasting_blood_sugar_gt_120_mg_per_dl == 1 ~ "yes")) %>% 
  ## IMPORTANT NOTE ON EKG - VALUE 1 LINKED WITH ST elevation or depression greater than 0.05 mV. Value 2 uses Estes' criteria measurement.
  mutate(CATresting_ekg_results = case_when(resting_ekg_results == 0 ~ "normal",
                                            resting_ekg_results == 1 ~ "ST-T_wave_abnormality",
                                            resting_ekg_results == 2 ~ "likely_left_ventricular_hypertrophy" )) %>%
  mutate(CATexercise_induced_angina = case_when(exercise_induced_angina == 0 ~ "no",
                                                exercise_induced_angina == 1 ~ "yes")) %>%
  mutate(CATMaxHeartRateAchieved = ifelse(max_heart_rate_achieved < (206.9-(0.67 * age)),"Below_MHR_ForAge",
                                          "AtorAbove_MHR_ForAge" )) %>%
  mutate(CATOldpeakSTDepression = ifelse(oldpeak_eq_st_depression <= 2.0, "minor_ST_depression",
                                         "significant_ST_depression" )) %>%
  select(-NUMthal,
         -sex,
         -chest_pain_type,
         -slope_of_peak_exercise_st_segment,
         -fasting_blood_sugar_gt_120_mg_per_dl,
         -resting_ekg_results,
         -exercise_induced_angina) %>%
  
  dplyr::rename(TARGET_heart_disease_present   = heart_disease_present,
                CATthal                        = thal, 
                NUMresting_blood_pressure      = resting_blood_pressure,
                NUMnum_major_vessels           = num_major_vessels,
                NUMserum_cholesterol_mg_per_dl = serum_cholesterol_mg_per_dl,
                NUMoldpeak_eq_st_depression    = oldpeak_eq_st_depression,
                NUMage                         = age,
                NUMmax_heart_rate_achieved     = max_heart_rate_achieved) %>%
  #Convert the integer/logical target variable to factor
  mutate_at(.vars = vars(starts_with("CAT"), "TARGET_heart_disease_present"), #more categorical fields can be added here to convert in bulk,
            .funs = funs(as.factor(.)))


#### FEATURE SELECTION ##########
glimpse(wrk.combined_datatypecleaned)
wrk.combined_datatypecleaned <- select(wrk.combined_datatypecleaned, patient_id, TARGET_heart_disease_present, CATDataSetOrigin,
                                       CATthal,CATchest_pain_type,  NUMoldpeak_eq_st_depression, NUMnum_major_vessels, NUMmax_heart_rate_achieved,
                                       CATsex, CATexercise_induced_angina, CATslope_of_peak_exercise_st_segment,
                                       NUMage)
features_selected = c("CATthal","CATchest_pain_type",  "NUMoldpeak_eq_st_depression", "NUMnum_major_vessels", "NUMmax_heart_rate_achieved",
                      "CATsex", "CATexercise_induced_angina", "CATslope_of_peak_exercise_st_segment",
                      "NUMage")
#################################


library(skimr)
skim(wrk.combined_datatypecleaned)
# sparkline graphs for numerics!

# DataExplorer
create_report(wrk.combined_datatypecleaned, output_file = "DataExplorer_Report_FullDset.html", output_dir = F("/Docs"),
              y = "TARGET_heart_disease_present", config = config,
              html_document(toc = TRUE, toc_depth = 6, theme = "flatly"))


# 1. Target class distribution
# take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count and as a percentage.
# summarize the class distribution
percentage <- prop.table(table(wrk.combined_datatypecleaned$TARGET_heart_disease_present)) * 100
cbind(freq=table(wrk.combined_datatypecleaned$TARGET_heart_disease_present), percentage = percentage)

# 2. Statistical summaries
# Now finally, we can take a look at a summary of each attribute in relation to the target variable
library(skimr)
library(dplyr)
mydata <- group_by(wrk.combined_datatypecleaned, TARGET_heart_disease_present) %>%
  skim()
mydata

# 3. Simple data visualisations
#We now have a basic idea about the data. We need to extend that with some visualizations.
#We are going to look at two types of plots:

# 1. Univariate plots to better understand each attribute.
# 2. Multivariate plots to better understand the relationships between attributes.
# split input and output
x <- select(wrk.combined_datatypecleaned, -patient_id, -TARGET_heart_disease_present, -CATDataSetOrigin )
x_num <- select(wrk.combined_datatypecleaned,(starts_with("NUM")))
x_cat <- select(wrk.combined_datatypecleaned,(starts_with("CAT")),"TARGET_heart_disease_present",  -patient_id, -CATDataSetOrigin)
y <- select(wrk.combined_datatypecleaned, TARGET_heart_disease_present)

# We can also create a barplot of
# the target class variable to get a graphical representation of the class distribution
# (generally uninteresting in this case because they are close to being balanced).
# barplot for class breakdown
plot(y, main = "Target class distribution")


# Check association between Age and Max Heart rate. if we see association here, not
# a great idea to go creating features based off this.
ggplot(wrk.combined_datatypecleaned, aes(x=NUMmax_heart_rate_achieved, NUMage)) + 
  geom_point(aes(colour = factor(CATsex)), size = 3) +
  geom_smooth(method = 'lm') +
  ggtitle("Checking association between age and max heart rate")

#Chi square test between Age and Max HR -- looking for P value less than 0.05 and a large X-squred chi sq value.
#H0: Null hypothesis: The The two variables are independent. P > 0.05
#H1: Rejecting null hypothesis: The two variables are related. P < 0.05
chisq.test(wrk.combined_datatypecleaned$NUMage, wrk.combined_datatypecleaned$NUMmax_heart_rate_achieved)
# X-squared = 3845, df = 3560, p-value = 0.0004851:: Age v max HR are related.

chisq.test(wrk.combined_datatypecleaned$NUMage, wrk.combined_datatypecleaned$NUMnum_major_vessels) # vert weak, but related
chisq.test(wrk.combined_datatypecleaned$NUMage, wrk.combined_datatypecleaned$NUMoldpeak_eq_st_depression)

# Try mass Chi Square test (see 99_Functions_MassChi.R)
funMassChi(x_num)
Ergebnis
# delFirst parameter can delete the first n columns. So if you have an count index or something you dont want to test.
# Can also specify a path for XL output
# What else is age associated with the age variable??? Check the correlation plot.


# Multivariate plots
#Now we can look at the interactions between the variables.
#First let’s look at scatterplots of all pairs of attributes and color the points by class.
#In addition, because the scatterplots show that points for each class are generally separate,
#we can draw ellipses around them.

# scatterplot matrix with ellipses
library(caret)
library(lattice)
library(AppliedPredictiveModeling)

transparentTheme(trans = .4)
caret::featurePlot(x = x_num, 
                   y = y$TARGET_heart_disease_present, 
                   plot = "ellipse", #alt "pairs"
                   ## Add a key at the top (two levels)
                   auto.key = list(columns = 2))


# Overlayed density plots
transparentTheme(trans = .9)
featurePlot(x = x_num,
            y = y$TARGET_heart_disease_present,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(3, 2), 
            auto.key = list(columns = 2))

# box and whisker plots for each attribute (numeric)
featurePlot(x = x_num, 
            y = y$TARGET_heart_disease_present, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(3,2 ), 
            auto.key = list(columns = 2))

# Categorical/Factor- grouped bar plots - checking frequency counts
ggplot(x_cat, aes(x=CATthal, fill=TARGET_heart_disease_present)) + 
  geom_bar(position="dodge", stat="count")
ggplot(x_cat, aes(x=CATsex, fill=TARGET_heart_disease_present)) + 
  geom_bar(position="dodge", stat="count")
ggplot(x_cat, aes(x=CATslope_of_peak_exercise_st_segment, fill=TARGET_heart_disease_present)) + 
  geom_bar(position="dodge", stat="count")
ggplot(x_cat, aes(x=CATchest_pain_type, fill=TARGET_heart_disease_present)) + 
  geom_bar(position="dodge", stat="count")
ggplot(x_cat, aes(x=CATexercise_induced_angina, fill=TARGET_heart_disease_present)) + 
  geom_bar(position="dodge", stat="count")


# Create a MOSIAC PLOT OF CHAR/FACTOR VS TARGET
#See code 99_Functions
#plot.new()
#dev.off()
x_cat_p1 <- select(x_cat, TARGET_heart_disease_present, CATthal, CATsex)
x_cat_p2 <- select(x_cat, TARGET_heart_disease_present, CATslope_of_peak_exercise_st_segment)
x_cat_p3 <- select(x_cat, TARGET_heart_disease_present, CATchest_pain_type)
x_cat_p4 <- select(x_cat, TARGET_heart_disease_present, CATexercise_induced_angina)
multiplot(x_cat_p1, 'TARGET_heart_disease_present')
multiplot(x_cat_p2, 'TARGET_heart_disease_present')
multiplot(x_cat_p3, 'TARGET_heart_disease_present')
multiplot(x_cat_p4, 'TARGET_heart_disease_present')

# 3.1 Correlation plot (small number of original numeric vars):
library(corrplot)
cor(x_num) # print correlation matrix results first, then visually plot.
corrplot.mixed(cor(x_num), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order = "hclust", hclust.method="complete")

# 3.2 TSNE: Are the features seperable/separatable/distinct enough or too noisy to find a clear signal for use in prediction?
# REF:https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
# This step is purely observational. If we find distinct groups emerging, then we know
# with a degree of confidence that our set of features will be sufficient to predict with.
# If the visual looks noisy, perhaps we can invest in dropping some less contributory features, then attempt prediction.
# T-SNE (using instead of PCA)
library(Rtsne)
## Curating the database for analysis with both t-SNE and PCA
#Labels<-train$label
#train$label<-as.factor(train$label)
## for plotting
colors = rainbow(length(unique(y$TARGET_heart_disease_present)))
names(colors) = unique(y$TARGET_heart_disease_present)

## Executing the algorithm on curated data
tsne <- Rtsne(x, dims = 2, perplexity=9, verbose=TRUE, max_iter = 500)
exeTimeTsne <- system.time(Rtsne(x, dims = 2, perplexity=9, verbose=TRUE, max_iter = 500))

## Plotting
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=y$TARGET_heart_disease_present, col=colors[y$TARGET_heart_disease_present])


# 4. Summary Report
# DescTools 
#wrd_01 <- GetNewWrd()
# all the features/attributes versus response (. ~ target) - A different perspective.
wrk.forDescTools <- wrk.combined_datatypecleaned %>%
  select(-patient_id, -CATDataSetOrigin)
#Desc(wrd = wrd_01, data = wrk.forDescTools, . ~ TARGET_heart_disease_present , plotit = TRUE, verbose = 3)

#SRC File: D:\TAC Data Science\04 Projects\Kaggle Competitions\Driven Data Org - Predicting Heart Disease\R\Docs\DescTools - Describe wrk-combined_datatypecleaned - All vs Target.docx
#Convert to PDF.
#wrd_01$ActiveDocument()$SaveAs2(FileName=F("/Data/R_DescTools_HeartDisease_vs_Features.docx"))

# Find Zero Variance and Near Zero Variance columns and remove from combined set
nzv <- caret::nearZeroVar(wrk.combined_datatypecleaned, saveMetrics = TRUE)
cols_zeroVar <- rownames(nzv[which(nzv$zeroVar == TRUE), ])
cols_nearzeroVar <- rownames(nzv[which(nzv$nzv == TRUE), ])
`%ni%` <- Negate(`%in%`)
wrk.combined_datatypecleaned <- subset(wrk.combined_datatypecleaned, select = names(wrk.combined_datatypecleaned) %ni% cols_zeroVar)

# Make sure categoricals are factors
wrk.combined_datatypecleaned <- wrk.combined_datatypecleaned %>%
  mutate_at( #Convert the integer/logical target variable to factor
    .vars = vars(starts_with("CAT"),"TARGET_heart_disease_present"), #more categorical fields can be added here to convert in bulk,
    .funs = funs(as.factor(.))
  )

# One-hot enccode (create dummy) variables - all except the target class and dataset origin flag.
wrk.combined_datatypecleaned <- mlr::createDummyFeatures(
  wrk.combined_datatypecleaned, target = "TARGET_heart_disease_present",
  cols = c("CATthal","CATchest_pain_type", "CATsex", "CATexercise_induced_angina", "CATslope_of_peak_exercise_st_segment")
  #add in other factor variables here for bulk one-hot encoding.
)

summarizeColumns(wrk.combined_datatypecleaned) %>%
  kable(digits = 2)

#check correlations:
corrplot_num <- select_if(wrk.combined_datatypecleaned, is.numeric) #only numerics, target not included.
corrplot.mixed(cor(corrplot_num), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
#Check TSNE again
tsne <- Rtsne(corrplot_num, dims = 2, perplexity=18, verbose=TRUE, max_iter = 500)
exeTimeTsne <- system.time(Rtsne(corrplot_num, dims = 2, perplexity=18, verbose=TRUE, max_iter = 500))
## Plotting
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=y$TARGET_heart_disease_present, col=colors[y$TARGET_heart_disease_present])


# check for nzv again: resting_ekg_results.1  is NEAR ZERO VARIANCE
nzv_ohe <- nearZeroVar(wrk.combined_datatypecleaned, saveMetrics = TRUE)
cols_ohe <- rownames(nzv_ohe[which(nzv_ohe$zeroVar == TRUE), ])
`%ni%` <- Negate(`%in%`)
wrk.combined_datatypecleaned <- subset(wrk.combined_datatypecleaned, select = names(wrk.combined_datatypecleaned) %ni% cols_ohe)

# Check the dataset for our newly created OHE vars
str(wrk.combined_datatypecleaned)

# TEST SETUP SET FOR FINAL PREDICTIONS AND SUBMISSION (Capture test data and test IDs)
mod.test <- wrk.combined_datatypecleaned %>% 
  filter(CATDataSetOrigin == "Testing") %>%
  select(-CATDataSetOrigin) # and keep patient id for later

mod.test_DATA <- select(mod.test, -patient_id, -TARGET_heart_disease_present )
mod.test_ID   <- mod.test$patient_id

str(mod.test_DATA)
str(mod.test_ID)