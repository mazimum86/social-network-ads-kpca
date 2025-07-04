# =============================================
# Logistic Regression on Social Network Ads
# Using Kernel PCA
# =============================================

# ========== Import Dataset ==========
dataset <- read.csv('../data/Social_network_Ads.csv')
# Columns: Age, EstimatedSalary, Purchased

# ========== Train/Test Split ==========
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
dataset_train <- subset(dataset, split == TRUE)
dataset_test  <- subset(dataset, split == FALSE)

# ========== Feature Scaling ==========
dataset_train[, 1:2] <- scale(dataset_train[, 1:2])
dataset_test[, 1:2]  <- scale(dataset_test[, 1:2])

# ========== Apply Kernel PCA ==========
library(kernlab)
kpca <- kpca(~ ., data = dataset_train[, 1:2], kernel = 'rbfdot', features = 2)

# Transform training and test sets
dataset_train_kpca <- as.data.frame(predict(kpca, dataset_train[, 1:2]))
dataset_test_kpca  <- as.data.frame(predict(kpca, dataset_test[, 1:2]))

# Append labels
dataset_train <- cbind(dataset_train_kpca, Purchased = dataset_train$Purchased)
dataset_test  <- cbind(dataset_test_kpca, Purchased = dataset_test$Purchased)

# ========== Fit Logistic Regression ==========
classifier <- glm(
  formula = Purchased ~ .,
  family = binomial,
  data = dataset_train
)

# ========== Make Predictions ==========
prob_pred <- predict(classifier, type = 'response', newdata = dataset_test)
y_pred <- as.numeric(prob_pred > 0.5)

# ========== Confusion Matrix ==========
cm <- table(Actual = dataset_test$Purchased, Predicted = y_pred)
print(cm)

# ========== Visualization Helper Function ==========
plot_decision_boundary <- function(set, title) {
  X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
  X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
  grid <- expand.grid(V1 = X1, V2 = X2)
  
  grid_pred <- as.numeric(predict(classifier, type = 'response', newdata = grid) > 0.5)
  
  # Plot background decision region
  image(
    X1, X2,
    matrix(grid_pred, length(X1), length(X2)),
    col = c('tomato', 'springgreen3'),
    xlab = 'KPC1', ylab = 'KPC2',
    main = title,
    useRaster = TRUE
  )
  
  # Overlay points
  points(grid, pch = '.', col = ifelse(grid_pred == 1, 'springgreen3', 'tomato'))
  points(set[, 1:2], pch = 21, bg = ifelse(set$Purchased == 1, 'green4', 'red3'))
}

# ========== Visualize Training Set ==========
plot_decision_boundary(dataset_train, title = 'Logistic Regression (Training Set)')

# ========== Visualize Test Set ==========
plot_decision_boundary(dataset_test, title = 'Logistic Regression (Test Set)')
