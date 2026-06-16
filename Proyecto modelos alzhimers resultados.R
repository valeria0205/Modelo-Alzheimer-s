# =========================================================
# APÉNDICE METODOLÓGICO
# Proyecto: Predicción de diagnóstico de Alzheimer
# Comparación entre variables directas e indirectas
# =========================================================


# =========================================================
# 1. PAQUETES
# Se cargan todos los paquetes requeridos al inicio.
# Nota: install.packages() debe ejecutarse una sola vez,
# fuera del script final de análisis.
# =========================================================
library(DBI)
library(odbc)
library(dplyr)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(nnet)


# =========================================================
# 2. CONEXIÓN A SQL SERVER Y CARGA DE DATOS
# Se importa la tabla consolidada desde SQL Server.
# =========================================================
con <- DBI::dbConnect(
  odbc::odbc(),
  Driver = "SQL Server",
  Server = "DESKTOP-SQQSJLC\\SQLEXPRESS",
  Database = "Biotec2026",
  Trusted_Connection = "Yes"
)

# Verificar conexión
DBI::dbIsValid(con)

# Cargar tabla principal
df_base <- DBI::dbGetQuery(con, "SELECT * FROM dbo.patients_combined")

# Revisión inicial
dim(df_base)
names(df_base)
head(df_base)
table(df_base$Diagnosis)


# =========================================================
# 3. LIMPIEZA BÁSICA
# Se elimina cualquier duplicado por PatientID
# y se asegura que Diagnosis sea una variable categórica.
# =========================================================
df_base <- df_base %>%
  distinct(PatientID, .keep_all = TRUE)

df_base$Diagnosis <- factor(df_base$Diagnosis, levels = c(0, 1))

# Revisiones básicas
sum(duplicated(df_base$PatientID))
sum(is.na(df_base$Diagnosis))
table(df_base$Diagnosis)


# =========================================================
# 4. DEFINICIÓN DE BLOQUES DE VARIABLES
# Indirecto = demográficas + factores clínicos indirectos
# Directo = pruebas cognitivas, funcionales y síntomas
# =========================================================
vars_indirectas <- c(
  "PatientID", "Diagnosis",
  "Age", "Gender", "Ethnicity", "EducationLevel",
  "BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity",
  "DietQuality", "SleepQuality", "FamilyHistoryAlzheimers",
  "CardiovascularDisease", "Diabetes", "Depression",
  "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP",
  "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
  "CholesterolTriglycerides"
)

vars_directas <- c(
  "PatientID", "Diagnosis",
  "MMSE", "FunctionalAssessment", "MemoryComplaints",
  "BehavioralProblems", "ADL", "Confusion", "Disorientation",
  "PersonalityChanges", "DifficultyCompletingTasks",
  "Forgetfulness"
)

# Verificar disponibilidad de columnas
setdiff(vars_indirectas, names(df_base))
setdiff(vars_directas, names(df_base))

# Crear datasets
df_indirecto <- df_base %>% select(all_of(vars_indirectas))
df_directo   <- df_base %>% select(all_of(vars_directas))


# =========================================================
# 5. DIVISIÓN TRAIN / TEST
# Se utiliza un split estratificado 70/30 para mantener
# proporciones similares de Diagnosis en ambos subconjuntos.
# =========================================================
set.seed(123)

idx_train <- createDataPartition(df_base$Diagnosis, p = 0.70, list = FALSE)

train_ids <- df_base$PatientID[idx_train]
test_ids  <- df_base$PatientID[-idx_train]

# Verificación del split
length(train_ids)
length(test_ids)
length(intersect(train_ids, test_ids))   # Debe ser 0

prop.table(table(df_base$Diagnosis))
prop.table(table(df_base$Diagnosis[idx_train]))
prop.table(table(df_base$Diagnosis[-idx_train]))


# =========================================================
# 6. CREACIÓN DE CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
# Se aplica el mismo split a los bloques directo e indirecto.
# =========================================================
train_indirecto <- df_indirecto %>% filter(PatientID %in% train_ids)
test_indirecto  <- df_indirecto %>% filter(PatientID %in% test_ids)

train_directo <- df_directo %>% filter(PatientID %in% train_ids)
test_directo  <- df_directo %>% filter(PatientID %in% test_ids)

# Eliminar PatientID antes del modelado
train_indirecto_model <- train_indirecto %>% select(-PatientID)
test_indirecto_model  <- test_indirecto %>% select(-PatientID)

train_directo_model <- train_directo %>% select(-PatientID)
test_directo_model  <- test_directo %>% select(-PatientID)

# Asegurar clasificación binaria
train_indirecto_model$Diagnosis <- factor(train_indirecto_model$Diagnosis, levels = c(0, 1))
test_indirecto_model$Diagnosis  <- factor(test_indirecto_model$Diagnosis,  levels = c(0, 1))

train_directo_model$Diagnosis <- factor(train_directo_model$Diagnosis, levels = c(0, 1))
test_directo_model$Diagnosis  <- factor(test_directo_model$Diagnosis,  levels = c(0, 1))


# =========================================================
# 7. FUNCIÓN AUXILIAR DE EVALUACIÓN
# Resume métricas principales para cada modelo.
# =========================================================
evaluar_modelo <- function(real, pred_clase, pred_prob) {
  real <- factor(real, levels = c(0, 1))
  pred_clase <- factor(pred_clase, levels = c(0, 1))
  
  cm <- confusionMatrix(pred_clase, real, positive = "1")
  roc_obj <- roc(real, pred_prob)
  
  list(
    confusion = cm,
    accuracy = as.numeric(cm$overall["Accuracy"]),
    sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    specificity = as.numeric(cm$byClass["Specificity"]),
    auc = as.numeric(auc(roc_obj))
  )
}


# =========================================================
# 8. REGRESIÓN LOGÍSTICA
# Modelo lineal base para clasificación binaria.
# =========================================================
modelo_logit_indirecto <- glm(
  Diagnosis ~ .,
  data = train_indirecto_model,
  family = binomial
)

modelo_logit_directo <- glm(
  Diagnosis ~ .,
  data = train_directo_model,
  family = binomial
)

summary(modelo_logit_indirecto)
summary(modelo_logit_directo)

# Predicción en test
prob_logit_indirecto <- predict(modelo_logit_indirecto, newdata = test_indirecto_model, type = "response")
prob_logit_directo   <- predict(modelo_logit_directo,   newdata = test_directo_model,   type = "response")

pred_logit_indirecto <- ifelse(prob_logit_indirecto > 0.5, 1, 0)
pred_logit_directo   <- ifelse(prob_logit_directo > 0.5, 1, 0)

res_logit_indirecto <- evaluar_modelo(test_indirecto_model$Diagnosis, pred_logit_indirecto, prob_logit_indirecto)
res_logit_directo   <- evaluar_modelo(test_directo_model$Diagnosis,   pred_logit_directo,   prob_logit_directo)


# =========================================================
# 9. ÁRBOL DE DECISIÓN
# Modelo interpretable basado en reglas de partición.
# =========================================================
modelo_tree_indirecto <- rpart(
  Diagnosis ~ .,
  data = train_indirecto_model,
  method = "class"
)

modelo_tree_directo <- rpart(
  Diagnosis ~ .,
  data = train_directo_model,
  method = "class"
)

printcp(modelo_tree_indirecto)
printcp(modelo_tree_directo)

# Visualización
rpart.plot(modelo_tree_indirecto, main = "Árbol indirecto")
rpart.plot(modelo_tree_directo, main = "Árbol directo")

# Importancia de variables
modelo_tree_indirecto$variable.importance
modelo_tree_directo$variable.importance

# Predicción en test
pred_tree_indirecto <- predict(modelo_tree_indirecto, newdata = test_indirecto_model, type = "class")
pred_tree_directo   <- predict(modelo_tree_directo,   newdata = test_directo_model,   type = "class")

prob_tree_indirecto <- predict(modelo_tree_indirecto, newdata = test_indirecto_model, type = "prob")[, "1"]
prob_tree_directo   <- predict(modelo_tree_directo,   newdata = test_directo_model,   type = "prob")[, "1"]

res_tree_indirecto <- evaluar_modelo(test_indirecto_model$Diagnosis, pred_tree_indirecto, prob_tree_indirecto)
res_tree_directo   <- evaluar_modelo(test_directo_model$Diagnosis,   pred_tree_directo,   prob_tree_directo)


# =========================================================
# 10. RANDOM FOREST
# Ensamble de árboles. Diagnosis debe ser factor para que
# el modelo sea de clasificación y no de regresión.
# =========================================================
set.seed(123)

modelo_rf_indirecto <- randomForest(
  Diagnosis ~ .,
  data = train_indirecto_model,
  ntree = 500,
  importance = TRUE
)

modelo_rf_directo <- randomForest(
  Diagnosis ~ .,
  data = train_directo_model,
  ntree = 500,
  importance = TRUE
)

modelo_rf_indirecto
modelo_rf_directo

# Importancia de variables
importance(modelo_rf_indirecto)
importance(modelo_rf_directo)

varImpPlot(modelo_rf_indirecto, main = "Importancia RF indirecto")
varImpPlot(modelo_rf_directo, main = "Importancia RF directo")

# Predicción en test
pred_rf_indirecto <- predict(modelo_rf_indirecto, newdata = test_indirecto_model, type = "response")
pred_rf_directo   <- predict(modelo_rf_directo,   newdata = test_directo_model,   type = "response")

prob_rf_indirecto <- predict(modelo_rf_indirecto, newdata = test_indirecto_model, type = "prob")[, "1"]
prob_rf_directo   <- predict(modelo_rf_directo,   newdata = test_directo_model,   type = "prob")[, "1"]

res_rf_indirecto <- evaluar_modelo(test_indirecto_model$Diagnosis, pred_rf_indirecto, prob_rf_indirecto)
res_rf_directo   <- evaluar_modelo(test_directo_model$Diagnosis,   pred_rf_directo,   prob_rf_directo)


# =========================================================
# 11. SUPPORT VECTOR MACHINE (SVM)
# Clasificador no lineal con kernel radial.
# =========================================================
set.seed(123)

modelo_svm_indirecto <- svm(
  Diagnosis ~ .,
  data = train_indirecto_model,
  kernel = "radial",
  scale = TRUE,
  probability = TRUE
)

modelo_svm_directo <- svm(
  Diagnosis ~ .,
  data = train_directo_model,
  kernel = "radial",
  scale = TRUE,
  probability = TRUE
)

modelo_svm_indirecto
modelo_svm_directo

# Predicción en test
pred_svm_indirecto <- predict(modelo_svm_indirecto, newdata = test_indirecto_model)
pred_svm_directo   <- predict(modelo_svm_directo,   newdata = test_directo_model)

prob_svm_indirecto <- attr(
  predict(modelo_svm_indirecto, newdata = test_indirecto_model, probability = TRUE),
  "probabilities"
)[, "1"]

prob_svm_directo <- attr(
  predict(modelo_svm_directo, newdata = test_directo_model, probability = TRUE),
  "probabilities"
)[, "1"]

res_svm_indirecto <- evaluar_modelo(test_indirecto_model$Diagnosis, pred_svm_indirecto, prob_svm_indirecto)
res_svm_directo   <- evaluar_modelo(test_directo_model$Diagnosis,   pred_svm_directo,   prob_svm_directo)


# =========================================================
# 12. RED NEURONAL
# Red simple con una capa oculta de 5 nodos.
# =========================================================
set.seed(123)

modelo_nn_indirecto <- nnet(
  Diagnosis ~ .,
  data = train_indirecto_model,
  size = 5,
  decay = 0.01,
  maxit = 300,
  trace = FALSE
)

modelo_nn_directo <- nnet(
  Diagnosis ~ .,
  data = train_directo_model,
  size = 5,
  decay = 0.01,
  maxit = 300,
  trace = FALSE
)

modelo_nn_indirecto
modelo_nn_directo

# Predicción en test
prob_nn_indirecto <- predict(modelo_nn_indirecto, newdata = test_indirecto_model, type = "raw")
prob_nn_directo   <- predict(modelo_nn_directo,   newdata = test_directo_model,   type = "raw")

pred_nn_indirecto <- ifelse(prob_nn_indirecto > 0.5, 1, 0)
pred_nn_directo   <- ifelse(prob_nn_directo > 0.5, 1, 0)

res_nn_indirecto <- evaluar_modelo(test_indirecto_model$Diagnosis, pred_nn_indirecto, as.vector(prob_nn_indirecto))
res_nn_directo   <- evaluar_modelo(test_directo_model$Diagnosis,   pred_nn_directo,   as.vector(prob_nn_directo))


# =========================================================
# 13. TABLA FINAL COMPARATIVA
# Resume el desempeño de todos los modelos.
# =========================================================
resultados <- data.frame(
  Modelo = c(
    "Logística indirecto", "Logística directo",
    "Árbol indirecto", "Árbol directo",
    "RF indirecto", "RF directo",
    "SVM indirecto", "SVM directo",
    "NN indirecto", "NN directo"
  ),
  Accuracy = c(
    res_logit_indirecto$accuracy, res_logit_directo$accuracy,
    res_tree_indirecto$accuracy, res_tree_directo$accuracy,
    res_rf_indirecto$accuracy, res_rf_directo$accuracy,
    res_svm_indirecto$accuracy, res_svm_directo$accuracy,
    res_nn_indirecto$accuracy, res_nn_directo$accuracy
  ),
  Sensitivity = c(
    res_logit_indirecto$sensitivity, res_logit_directo$sensitivity,
    res_tree_indirecto$sensitivity, res_tree_directo$sensitivity,
    res_rf_indirecto$sensitivity, res_rf_directo$sensitivity,
    res_svm_indirecto$sensitivity, res_svm_directo$sensitivity,
    res_nn_indirecto$sensitivity, res_nn_directo$sensitivity
  ),
  Specificity = c(
    res_logit_indirecto$specificity, res_logit_directo$specificity,
    res_tree_indirecto$specificity, res_tree_directo$specificity,
    res_rf_indirecto$specificity, res_rf_directo$specificity,
    res_svm_indirecto$specificity, res_svm_directo$specificity,
    res_nn_indirecto$specificity, res_nn_directo$specificity
  ),
  AUC = c(
    res_logit_indirecto$auc, res_logit_directo$auc,
    res_tree_indirecto$auc, res_tree_directo$auc,
    res_rf_indirecto$auc, res_rf_directo$auc,
    res_svm_indirecto$auc, res_svm_directo$auc,
    res_nn_indirecto$auc, res_nn_directo$auc
  )
)

resultados <- resultados %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

resultados


# =========================================================
# 14. CIERRE DE CONEXIÓN
# Se cierra la conexión con SQL Server al final del proceso.
# =========================================================
#DBI::dbDisconnect(con)