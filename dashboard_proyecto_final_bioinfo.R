# ============================================================
# DASHBOARD SIMPLE EN DOS PÁGINAS
#
# Página 1:
#   - Vista previa de la base
#   - Selección del identificador
#   - Selección de la variable resultado
#   - Selección de clase positiva
#   - Selección de variables directas e indirectas
#   - Botón para correr los modelos
#
# Página 2:
#   - Comparación de cinco modelos para cada bloque
#   - Accuracy, sensibilidad, especificidad y AUC
#   - Puntaje global simple
#   - Modelo ganador
#   - Matriz de confusión seleccionable con VN, FP, FN y VP
#   - Importancia predictiva separada para modelos directos e indirectos
# ============================================================


# ============================================================
# 0. PAQUETES
# ============================================================

# Instalar una sola vez, si hace falta:
#
# install.packages(c(
#   "shiny", "shinydashboard", "DBI", "odbc", "dplyr",
#   "caret", "pROC", "rpart", "randomForest",
#   "e1071", "nnet", "ggplot2", "DT"
# ))

library(shiny)
library(shinydashboard)
library(DBI)
library(odbc)
library(dplyr)
library(caret)
library(pROC)
library(rpart)
library(randomForest)
library(e1071)
library(nnet)
library(ggplot2)
library(DT)


# ============================================================
# 1. CARGAR LA BASE DESDE SQL SERVER
# ============================================================

con <- DBI::dbConnect(
  odbc::odbc(),
  Driver = "SQL Server",
  Server = "DESKTOP-SQQSJLC\\SQLEXPRESS",
  Database = "Biotec2026",
  Trusted_Connection = "Yes"
)

datos <- DBI::dbGetQuery(
  con,
  "SELECT * FROM dbo.patients_combined"
)

DBI::dbDisconnect(con)


# ============================================================
# 2. SELECCIONES PREDETERMINADAS DEL PROYECTO
# ============================================================

directas_default <- c(
  "MMSE",
  "FunctionalAssessment",
  "MemoryComplaints",
  "BehavioralProblems",
  "ADL",
  "Confusion",
  "Disorientation",
  "PersonalityChanges",
  "DifficultyCompletingTasks",
  "Forgetfulness"
)

directas_default <- intersect(
  directas_default,
  names(datos)
)

indirectas_default <- setdiff(
  names(datos),
  c(
    "PatientID",
    "Diagnosis",
    directas_default
  )
)


# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

# Convierte texto, variables lógicas y variables numéricas
# con pocas categorías en factores.
convertir_predictores <- function(df) {

  df[] <- lapply(
    df,
    function(x) {

      if (is.character(x) || is.logical(x)) {
        return(factor(x))
      }

      if (is.numeric(x) && dplyr::n_distinct(x) <= 10) {
        return(factor(x))
      }

      x
    }
  )

  df
}


# Prepara un bloque de variables.
# La codificación, eliminación de columnas casi constantes,
# centrado y escalado se aprenden únicamente con training.
preparar_bloque <- function(
  train_data,
  test_data,
  predictores
) {

  x_train_original <- train_data %>%
    select(all_of(predictores))

  x_test_original <- test_data %>%
    select(all_of(predictores))

  codificador <- caret::dummyVars(
    ~ .,
    data = x_train_original,
    fullRank = TRUE
  )

  x_train <- as.data.frame(
    predict(
      codificador,
      newdata = x_train_original
    )
  )

  x_test <- as.data.frame(
    predict(
      codificador,
      newdata = x_test_original
    )
  )

  # Asegura nombres válidos y consistentes.
  nombres_validos <- make.names(
    names(x_train),
    unique = TRUE
  )

  names(x_train) <- nombres_validos
  names(x_test) <- nombres_validos

  columnas_problematicas <- caret::nearZeroVar(
    x_train
  )

  if (length(columnas_problematicas) > 0) {

    columnas_utiles <- setdiff(
      names(x_train),
      names(x_train)[columnas_problematicas]
    )

  } else {

    columnas_utiles <- names(x_train)
  }

  if (length(columnas_utiles) == 0) {
    stop("El bloque seleccionado no contiene variables utilizables.")
  }

  x_train <- x_train[, columnas_utiles, drop = FALSE]
  x_test <- x_test[, columnas_utiles, drop = FALSE]

  preprocesador <- caret::preProcess(
    x_train,
    method = c("center", "scale")
  )

  x_train <- predict(
    preprocesador,
    newdata = x_train
  )

  x_test <- predict(
    preprocesador,
    newdata = x_test
  )

  list(
    x_train = x_train,
    x_test = x_test,
    y_train = train_data$Target,
    y_test = test_data$Target,
    x_test_original = x_test_original,
    predictores_originales = predictores,
    codificador = codificador,
    preprocesador = preprocesador,
    columnas_utiles = columnas_utiles
  )
}


# Aplica a nuevos datos el mismo procesamiento aprendido con training.
transformar_nuevos_datos <- function(
  preparacion,
  nuevos_datos
) {

  nuevos_dummy <- as.data.frame(
    predict(
      preparacion$codificador,
      newdata = nuevos_datos
    )
  )

  names(nuevos_dummy) <- make.names(
    names(nuevos_dummy),
    unique = TRUE
  )

  nuevos_dummy <- nuevos_dummy[
    ,
    preparacion$columnas_utiles,
    drop = FALSE
  ]

  predict(
    preparacion$preprocesador,
    newdata = nuevos_dummy
  )
}


# Calcula el AUC.
calcular_auc <- function(
  real,
  probabilidad
) {

  curva <- pROC::roc(
    response = real,
    predictor = probabilidad,
    levels = c("Negative", "Positive"),
    direction = "<",
    quiet = TRUE
  )

  as.numeric(
    pROC::auc(curva)
  )
}


# Calcula las cuatro métricas utilizadas en esta versión.
evaluar_modelo <- function(
  real,
  probabilidad
) {

  clase_predicha <- ifelse(
    probabilidad >= 0.5,
    "Positive",
    "Negative"
  )

  clase_predicha <- factor(
    clase_predicha,
    levels = c("Negative", "Positive")
  )

  real <- factor(
    real,
    levels = c("Negative", "Positive")
  )

  matriz <- caret::confusionMatrix(
    clase_predicha,
    real,
    positive = "Positive"
  )

  data.frame(
    Accuracy = as.numeric(
      matriz$overall["Accuracy"]
    ),
    Sensibilidad = as.numeric(
      matriz$byClass["Sensitivity"]
    ),
    Especificidad = as.numeric(
      matriz$byClass["Specificity"]
    ),
    AUC = calcular_auc(
      real,
      probabilidad
    ),

    # Filas = clase predicha; columnas = clase real.
    VN = as.numeric(
      matriz$table["Negative", "Negative"]
    ),
    FN = as.numeric(
      matriz$table["Negative", "Positive"]
    ),
    FP = as.numeric(
      matriz$table["Positive", "Negative"]
    ),
    VP = as.numeric(
      matriz$table["Positive", "Positive"]
    )
  )
}


# Obtiene la probabilidad de la clase positiva
# según el tipo de modelo.
predecir_probabilidad <- function(
  modelo,
  algoritmo,
  nuevos_datos
) {

  if (algoritmo == "Regresión logística") {

    return(
      as.numeric(
        predict(
          modelo,
          newdata = nuevos_datos,
          type = "response"
        )
      )
    )
  }

  if (algoritmo == "Árbol") {

    probabilidades <- predict(
      modelo,
      newdata = nuevos_datos,
      type = "prob"
    )

    return(
      as.numeric(
        probabilidades[, "Positive"]
      )
    )
  }

  if (algoritmo == "Random Forest") {

    probabilidades <- predict(
      modelo,
      newdata = nuevos_datos,
      type = "prob"
    )

    return(
      as.numeric(
        probabilidades[, "Positive"]
      )
    )
  }

  if (algoritmo == "SVM") {

    prediccion <- predict(
      modelo,
      newdata = nuevos_datos,
      probability = TRUE
    )

    probabilidades <- attr(
      prediccion,
      "probabilities"
    )

    if (!"Positive" %in% colnames(probabilidades)) {
      stop("El modelo SVM no devolvió la probabilidad de la clase positiva.")
    }

    return(
      as.numeric(
        probabilidades[, "Positive"]
      )
    )
  }

  if (algoritmo == "Red neuronal") {

    return(
      as.numeric(
        predict(
          modelo,
          as.matrix(nuevos_datos),
          type = "raw"
        )
      )
    )
  }

  stop("Algoritmo no reconocido.")
}


# Entrena los cinco modelos en un bloque de variables.
correr_modelos_bloque <- function(
  train_data,
  test_data,
  predictores,
  grupo
) {

  preparacion <- preparar_bloque(
    train_data = train_data,
    test_data = test_data,
    predictores = predictores
  )

  x_train <- preparacion$x_train
  x_test <- preparacion$x_test
  y_train <- preparacion$y_train
  y_test <- preparacion$y_test

  train_modelo <- data.frame(
    Target = y_train,
    x_train,
    check.names = FALSE
  )

  modelos <- list()
  resultados <- data.frame()

  # ----------------------------------------------------------
  # REGRESIÓN LOGÍSTICA
  # ----------------------------------------------------------

  modelo_logistica <- glm(
    Target ~ .,
    data = train_modelo,
    family = binomial
  )

  prob_logistica <- predecir_probabilidad(
    modelo_logistica,
    "Regresión logística",
    x_test
  )

  resultados <- bind_rows(
    resultados,
    data.frame(
      Modelo = "Regresión logística",
      Grupo = grupo,
      evaluar_modelo(
        y_test,
        prob_logistica
      )
    )
  )

  modelos[["Regresión logística"]] <- modelo_logistica


  # ----------------------------------------------------------
  # ÁRBOL DE DECISIÓN
  # ----------------------------------------------------------

  set.seed(123)

  modelo_arbol <- rpart::rpart(
    Target ~ .,
    data = train_modelo,
    method = "class"
  )

  prob_arbol <- predecir_probabilidad(
    modelo_arbol,
    "Árbol",
    x_test
  )

  resultados <- bind_rows(
    resultados,
    data.frame(
      Modelo = "Árbol",
      Grupo = grupo,
      evaluar_modelo(
        y_test,
        prob_arbol
      )
    )
  )

  modelos[["Árbol"]] <- modelo_arbol


  # ----------------------------------------------------------
  # RANDOM FOREST
  # ----------------------------------------------------------

  set.seed(123)

  modelo_rf <- randomForest::randomForest(
    Target ~ .,
    data = train_modelo,
    ntree = 500
  )

  prob_rf <- predecir_probabilidad(
    modelo_rf,
    "Random Forest",
    x_test
  )

  resultados <- bind_rows(
    resultados,
    data.frame(
      Modelo = "Random Forest",
      Grupo = grupo,
      evaluar_modelo(
        y_test,
        prob_rf
      )
    )
  )

  modelos[["Random Forest"]] <- modelo_rf


  # ----------------------------------------------------------
  # SUPPORT VECTOR MACHINE
  # ----------------------------------------------------------

  set.seed(123)

  modelo_svm <- e1071::svm(
    x = x_train,
    y = y_train,
    kernel = "radial",
    scale = FALSE,
    probability = TRUE
  )

  prob_svm <- predecir_probabilidad(
    modelo_svm,
    "SVM",
    x_test
  )

  resultados <- bind_rows(
    resultados,
    data.frame(
      Modelo = "SVM",
      Grupo = grupo,
      evaluar_modelo(
        y_test,
        prob_svm
      )
    )
  )

  modelos[["SVM"]] <- modelo_svm


  # ----------------------------------------------------------
  # RED NEURONAL
  # ----------------------------------------------------------

  set.seed(123)

  y_train_numerico <- as.numeric(
    y_train == "Positive"
  )

  modelo_nn <- nnet::nnet(
    x = as.matrix(x_train),
    y = y_train_numerico,
    size = 5,
    decay = 0.01,
    maxit = 300,
    entropy = TRUE,
    trace = FALSE,
    MaxNWts = 10000
  )

  prob_nn <- predecir_probabilidad(
    modelo_nn,
    "Red neuronal",
    x_test
  )

  resultados <- bind_rows(
    resultados,
    data.frame(
      Modelo = "Red neuronal",
      Grupo = grupo,
      evaluar_modelo(
        y_test,
        prob_nn
      )
    )
  )

  modelos[["Red neuronal"]] <- modelo_nn


  list(
    resultados = resultados,
    modelos = modelos,
    preparacion = preparacion
  )
}


# Calcula importancia por permutación para el modelo ganador.
# Una variable es más importante si al mezclarla disminuye más el AUC.
calcular_importancia <- function(
  modelo,
  algoritmo,
  preparacion,
  auc_original
) {

  importancia_lista <- lapply(
    preparacion$predictores_originales,
    function(variable) {

      datos_permutados <- preparacion$x_test_original

      set.seed(
        1000 + match(
          variable,
          preparacion$predictores_originales
        )
      )

      datos_permutados[[variable]] <- sample(
        datos_permutados[[variable]]
      )

      x_permutado <- transformar_nuevos_datos(
        preparacion,
        datos_permutados
      )

      prob_permutada <- predecir_probabilidad(
        modelo,
        algoritmo,
        x_permutado
      )

      auc_permutado <- calcular_auc(
        preparacion$y_test,
        prob_permutada
      )

      data.frame(
        Variable = variable,
        Importancia = auc_original - auc_permutado
      )
    }
  )

  bind_rows(
    importancia_lista
  ) %>%
    mutate(
      Importancia = pmax(
        Importancia,
        0
      )
    ) %>%
    arrange(
      desc(Importancia)
    )
}


# Ejecuta todo el análisis después de presionar el botón.
ejecutar_analisis <- function(
  datos_originales,
  key_field,
  outcome_field,
  positive_class,
  variables_directas,
  variables_indirectas
) {

  columnas_necesarias <- unique(
    c(
      key_field,
      outcome_field,
      variables_directas,
      variables_indirectas
    )
  )

  datos_analisis <- datos_originales[
    ,
    columnas_necesarias,
    drop = FALSE
  ]

  # Una observación por identificador.
  datos_analisis <- datos_analisis[
    !duplicated(
      datos_analisis[[key_field]]
    ),
    ,
    drop = FALSE
  ]

  # Versión simple: elimina filas incompletas.
  datos_analisis <- datos_analisis[
    complete.cases(datos_analisis),
    ,
    drop = FALSE
  ]

  clases <- unique(
    as.character(
      datos_analisis[[outcome_field]]
    )
  )

  if (length(clases) != 2) {
    stop("La variable resultado debe tener exactamente dos clases.")
  }

  if (!positive_class %in% clases) {
    stop("La clase positiva seleccionada no existe en la variable resultado.")
  }

  negative_class <- setdiff(
    clases,
    positive_class
  )

  datos_analisis$Target <- factor(
    as.character(
      datos_analisis[[outcome_field]]
    ),
    levels = c(
      negative_class,
      positive_class
    ),
    labels = c(
      "Negative",
      "Positive"
    )
  )

  todos_predictores <- unique(
    c(
      variables_directas,
      variables_indirectas
    )
  )

  datos_analisis[
    todos_predictores
  ] <- convertir_predictores(
    datos_analisis[
      todos_predictores
    ]
  )

  if (nrow(datos_analisis) < 30) {
    stop("Quedaron menos de 30 filas completas para el análisis.")
  }

  if (any(table(datos_analisis$Target) < 10)) {
    stop("Cada clase debe contener al menos 10 observaciones.")
  }

  set.seed(123)

  indice_train <- caret::createDataPartition(
    datos_analisis$Target,
    p = 0.70,
    list = FALSE
  )

  train_data <- datos_analisis[
    indice_train,
    ,
    drop = FALSE
  ]

  test_data <- datos_analisis[
    -indice_train,
    ,
    drop = FALSE
  ]

  resultado_directo <- correr_modelos_bloque(
    train_data = train_data,
    test_data = test_data,
    predictores = variables_directas,
    grupo = "Directo"
  )

  resultado_indirecto <- correr_modelos_bloque(
    train_data = train_data,
    test_data = test_data,
    predictores = variables_indirectas,
    grupo = "Indirecto"
  )

  resultados <- bind_rows(
    resultado_directo$resultados,
    resultado_indirecto$resultados
  ) %>%
    mutate(
      PuntajeGlobal = rowMeans(
        select(
          .,
          Accuracy,
          Sensibilidad,
          Especificidad,
          AUC
        ),
        na.rm = TRUE
      )
    ) %>%
    arrange(
      desc(PuntajeGlobal),
      desc(AUC)
    )

  mejor_modelo <- resultados %>%
    slice(1)

  mejor_directo <- resultados %>%
    filter(Grupo == "Directo") %>%
    arrange(
      desc(PuntajeGlobal),
      desc(AUC)
    ) %>%
    slice(1)

  mejor_indirecto <- resultados %>%
    filter(Grupo == "Indirecto") %>%
    arrange(
      desc(PuntajeGlobal),
      desc(AUC)
    ) %>%
    slice(1)

  list(
    resultados = resultados,
    mejor_modelo = mejor_modelo,
    mejor_directo = mejor_directo,
    mejor_indirecto = mejor_indirecto,
    resultado_directo = resultado_directo,
    resultado_indirecto = resultado_indirecto,
    n_filas = nrow(datos_analisis),
    n_train = nrow(train_data),
    n_test = nrow(test_data)
  )
}


# ============================================================
# 4. INTERFAZ
# ============================================================

ui <- dashboardPage(

  dashboardHeader(
    title = "Modelos predictivos"
  ),

  dashboardSidebar(
    sidebarMenuOutput(
      "menu_lateral"
    )
  ),

  dashboardBody(

    tabItems(

      # --------------------------------------------------------
      # PÁGINA 1: CONFIGURACIÓN
      # --------------------------------------------------------

      tabItem(
        tabName = "configuracion",

        fluidRow(
          box(
            width = 12,
            title = "Vista previa de la base de datos",
            status = "primary",
            solidHeader = TRUE,
            DTOutput(
              "preview_datos"
            )
          )
        ),

        fluidRow(

          box(
            width = 4,
            title = "Campos principales",
            status = "info",
            solidHeader = TRUE,

            selectInput(
              "key_field",
              "Key field o identificador",
              choices = names(datos),
              selected = if (
                "PatientID" %in% names(datos)
              ) {
                "PatientID"
              } else {
                names(datos)[1]
              }
            ),

            selectInput(
              "outcome_field",
              "Variable resultado o diagnóstico",
              choices = names(datos),
              selected = if (
                "Diagnosis" %in% names(datos)
              ) {
                "Diagnosis"
              } else {
                names(datos)[2]
              }
            ),

            uiOutput(
              "positive_class_ui"
            )
          ),

          box(
            width = 4,
            title = "Variables directas",
            status = "success",
            solidHeader = TRUE,

            selectizeInput(
              "direct_variables",
              "Seleccione las variables directas",
              choices = setdiff(
                names(datos),
                c(
                  "PatientID",
                  "Diagnosis"
                )
              ),
              selected = directas_default,
              multiple = TRUE
            )
          ),

          box(
            width = 4,
            title = "Variables indirectas",
            status = "warning",
            solidHeader = TRUE,

            selectizeInput(
              "indirect_variables",
              "Seleccione las variables indirectas",
              choices = setdiff(
                names(datos),
                c(
                  "PatientID",
                  "Diagnosis"
                )
              ),
              selected = indirectas_default,
              multiple = TRUE
            )
          )
        ),

        fluidRow(

          box(
            width = 8,
            title = "Resumen de la selección",
            status = "primary",
            solidHeader = TRUE,
            htmlOutput(
              "selection_summary"
            )
          ),

          box(
            width = 4,
            title = "Ejecutar análisis",
            status = "success",
            solidHeader = TRUE,

            actionButton(
              "run_models",
              "Correr modelos predictivos",
              icon = icon("play"),
              class = "btn-success btn-lg"
            ),

            br(),
            br(),

            htmlOutput(
              "run_status"
            )
          )
        )
      ),


      # --------------------------------------------------------
      # PÁGINA 2: RESULTADOS
      # --------------------------------------------------------

      tabItem(
        tabName = "resultados",

        fluidRow(
          valueBoxOutput(
            "value_rows",
            width = 3
          ),
          valueBoxOutput(
            "value_best_model",
            width = 5
          ),
          valueBoxOutput(
            "value_best_score",
            width = 4
          )
        ),

        fluidRow(

          box(
            width = 7,
            title = "Comparación de modelos",
            status = "primary",
            solidHeader = TRUE,
            plotOutput(
              "results_plot",
              height = 500
            )
          ),

          box(
            width = 5,
            title = "Interpretación del ganador",
            status = "info",
            solidHeader = TRUE,
            htmlOutput(
              "winner_text"
            )
          )
        ),

        fluidRow(

          box(
            width = 8,
            title = "Tabla de resultados",
            status = "primary",
            solidHeader = TRUE,
            DTOutput(
              "results_table"
            )
          ),

          box(
            width = 4,
            title = "Matriz de confusión",
            status = "info",
            solidHeader = TRUE,

            selectInput(
              "confusion_model",
              "Modelo y grupo",
              choices = character(0)
            ),

            tableOutput(
              "confusion_table"
            ),

            HTML(
              paste0(
                "<p><b>VN:</b> verdadero negativo</p>",
                "<p><b>FP:</b> falso positivo</p>",
                "<p><b>FN:</b> falso negativo</p>",
                "<p><b>VP:</b> verdadero positivo</p>"
              )
            )
          )
        ),

        fluidRow(

          box(
            width = 6,
            title = "Importancia predictiva — variables directas",
            status = "success",
            solidHeader = TRUE,

            selectInput(
              "importance_model_direct",
              "Modelo directo",
              choices = c(
                "Regresión logística",
                "Árbol",
                "Random Forest",
                "SVM",
                "Red neuronal"
              ),
              selected = "Random Forest"
            ),

            plotOutput(
              "importance_plot_direct",
              height = 470
            )
          ),

          box(
            width = 6,
            title = "Importancia predictiva — variables indirectas",
            status = "warning",
            solidHeader = TRUE,

            selectInput(
              "importance_model_indirect",
              "Modelo indirecto",
              choices = c(
                "Regresión logística",
                "Árbol",
                "Random Forest",
                "SVM",
                "Red neuronal"
              ),
              selected = "Random Forest"
            ),

            plotOutput(
              "importance_plot_indirect",
              height = 470
            )
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Cómo interpretar estas gráficas",
            status = "info",
            solidHeader = TRUE,
            HTML(
              paste0(
                "<p>La importancia se calcula por permutación. ",
                "Cada variable se mezcla de forma independiente y se vuelve ",
                "a calcular el AUC. Una caída mayor del AUC indica que el ",
                "modelo dependía más de esa variable para discriminar las clases.</p>",
                "<p>Por defecto se muestran los mejores modelos directo e ",
                "indirecto según el puntaje global. Los selectores permiten ",
                "examinar cualquiera de los otros algoritmos.</p>",
                "<p><b>Importante:</b> la importancia predictiva no implica causalidad.</p>"
              )
            )
          )
        )
      )
    )
  )
)


# ============================================================
# 5. SERVIDOR
# ============================================================

server <- function(
  input,
  output,
  session
) {

  analisis <- reactiveVal(
    NULL
  )


  # El menú de resultados aparece cuando termina el análisis.
  output$menu_lateral <- renderMenu({

    if (is.null(analisis())) {

      sidebarMenu(
        id = "tabs",

        menuItem(
          "Configurar análisis",
          tabName = "configuracion",
          icon = icon("sliders")
        )
      )

    } else {

      sidebarMenu(
        id = "tabs",

        menuItem(
          "Configurar análisis",
          tabName = "configuracion",
          icon = icon("sliders")
        ),

        menuItem(
          "Resultados",
          tabName = "resultados",
          icon = icon("dashboard")
        )
      )
    }
  })


  # Vista previa de las primeras 100 filas.
  output$preview_datos <- renderDT({

    datatable(
      head(
        datos,
        100
      ),
      rownames = FALSE,
      options = list(
        pageLength = 10,
        scrollX = TRUE
      )
    )
  })


  # Actualiza las variables disponibles cuando se cambia
  # el identificador o la variable resultado.
  observeEvent(
    list(
      input$key_field,
      input$outcome_field
    ),
    {

      req(
        input$key_field,
        input$outcome_field
      )

      opciones <- setdiff(
        names(datos),
        c(
          input$key_field,
          input$outcome_field
        )
      )

      seleccion_directa <- intersect(
        input$direct_variables,
        opciones
      )

      seleccion_indirecta <- intersect(
        input$indirect_variables,
        opciones
      )

      if (length(seleccion_directa) == 0) {
        seleccion_directa <- intersect(
          directas_default,
          opciones
        )
      }

      if (length(seleccion_indirecta) == 0) {
        seleccion_indirecta <- setdiff(
          opciones,
          seleccion_directa
        )
      }

      updateSelectizeInput(
        session,
        "direct_variables",
        choices = opciones,
        selected = seleccion_directa,
        server = TRUE
      )

      updateSelectizeInput(
        session,
        "indirect_variables",
        choices = opciones,
        selected = seleccion_indirecta,
        server = TRUE
      )
    },
    ignoreInit = FALSE
  )


  # Selector dinámico de clase positiva.
  output$positive_class_ui <- renderUI({

    req(
      input$outcome_field
    )

    clases <- unique(
      as.character(
        datos[[input$outcome_field]]
      )
    )

    clases <- clases[
      !is.na(clases)
    ]

    clase_default <- if (
      "1" %in% clases
    ) {
      "1"
    } else {
      clases[length(clases)]
    }

    selectInput(
      "positive_class",
      "Clase positiva",
      choices = clases,
      selected = clase_default
    )
  })


  output$selection_summary <- renderUI({

    HTML(
      paste0(
        "<p><b>Key field:</b> ",
        input$key_field,
        "</p>",

        "<p><b>Variable resultado:</b> ",
        input$outcome_field,
        "</p>",

        "<p><b>Variables directas:</b> ",
        length(
          input$direct_variables
        ),
        "</p>",

        "<p><b>Variables indirectas:</b> ",
        length(
          input$indirect_variables
        ),
        "</p>",

        "<p>Se correrán cinco algoritmos para cada bloque, ",
        "produciendo diez combinaciones en total.</p>"
      )
    )
  })


  # Ejecutar modelos.
  observeEvent(
    input$run_models,
    {

      req(
        input$key_field,
        input$outcome_field,
        input$positive_class,
        input$direct_variables,
        input$indirect_variables
      )

      if (input$key_field == input$outcome_field) {

        showNotification(
          "El identificador y la variable resultado deben ser diferentes.",
          type = "error"
        )

        return()
      }

      repetidas <- intersect(
        input$direct_variables,
        input$indirect_variables
      )

      if (length(repetidas) > 0) {

        showNotification(
          paste(
            "Estas variables están en ambos bloques:",
            paste(
              repetidas,
              collapse = ", "
            )
          ),
          type = "error",
          duration = NULL
        )

        return()
      }

      resultado <- tryCatch(

        withProgress(
          message = "Corriendo modelos predictivos",
          value = 0,
          {

            incProgress(
              0.1,
              detail = "Preparando datos"
            )

            resultado_temporal <- ejecutar_analisis(
              datos_originales = datos,
              key_field = input$key_field,
              outcome_field = input$outcome_field,
              positive_class = input$positive_class,
              variables_directas = input$direct_variables,
              variables_indirectas = input$indirect_variables
            )

            incProgress(
              0.9,
              detail = "Finalizando resultados"
            )

            resultado_temporal
          }
        ),

        error = function(e) {

          showNotification(
            paste(
              "No fue posible correr el análisis:",
              e$message
            ),
            type = "error",
            duration = NULL
          )

          NULL
        }
      )

      if (!is.null(resultado)) {

        analisis(
          resultado
        )

        updateSelectInput(
          session,
          "importance_model_direct",
          selected = resultado$mejor_directo$Modelo
        )

        updateSelectInput(
          session,
          "importance_model_indirect",
          selected = resultado$mejor_indirecto$Modelo
        )

        etiquetas_modelos <- paste(
          resultado$resultados$Modelo,
          resultado$resultados$Grupo,
          sep = " - "
        )

        valores_modelos <- paste(
          resultado$resultados$Modelo,
          resultado$resultados$Grupo,
          sep = ":::"
        )

        updateSelectInput(
          session,
          "confusion_model",
          choices = stats::setNames(
            valores_modelos,
            etiquetas_modelos
          ),
          selected = paste(
            resultado$mejor_modelo$Modelo,
            resultado$mejor_modelo$Grupo,
            sep = ":::"
          )
        )

        showNotification(
          "Los modelos finalizaron correctamente.",
          type = "message"
        )
      }
    }
  )


  output$run_status <- renderUI({

    if (is.null(analisis())) {

      HTML(
        "<p>Los resultados todavía no han sido generados.</p>"
      )

    } else {

      HTML(
        "<p><b>Análisis completado.</b> La página Resultados está habilitada en el menú.</p>"
      )
    }
  })


  # ----------------------------------------------------------
  # RESULTADOS
  # ----------------------------------------------------------

  output$value_rows <- renderValueBox({

    req(
      analisis()
    )

    valueBox(
      value = analisis()$n_filas,
      subtitle = "Filas analizadas",
      icon = icon("database"),
      color = "blue"
    )
  })


  output$value_best_model <- renderValueBox({

    req(
      analisis()
    )

    mejor <- analisis()$mejor_modelo

    valueBox(
      value = paste(
        mejor$Modelo,
        mejor$Grupo,
        sep = " - "
      ),
      subtitle = "Mejor modelo",
      icon = icon("trophy"),
      color = "green"
    )
  })


  output$value_best_score <- renderValueBox({

    req(
      analisis()
    )

    valueBox(
      value = round(
        analisis()$mejor_modelo$PuntajeGlobal,
        4
      ),
      subtitle = "Puntaje global",
      icon = icon("star"),
      color = "yellow"
    )
  })


  output$results_plot <- renderPlot({

    req(
      analisis()
    )

    resultados <- analisis()$resultados

    ggplot(
      resultados,
      aes(
        x = reorder(
          paste(
            Modelo,
            Grupo,
            sep = " - "
          ),
          PuntajeGlobal
        ),
        y = PuntajeGlobal,
        fill = Grupo
      )
    ) +
      geom_col() +
      coord_flip() +
      geom_text(
        aes(
          label = round(
            PuntajeGlobal,
            3
          )
        ),
        hjust = -0.1
      ) +
      ylim(
        0,
        1.05
      ) +
      labs(
        x = NULL,
        y = "Puntaje global",
        title = "Promedio de Accuracy, Sensibilidad, Especificidad y AUC"
      ) +
      theme_minimal()
  })


  output$results_table <- renderDT({

    req(
      analisis()
    )

    tabla <- analisis()$resultados %>%
      select(
        Modelo,
        Grupo,
        Accuracy,
        Sensibilidad,
        Especificidad,
        AUC,
        PuntajeGlobal
      ) %>%
      mutate(
        across(
          c(
            Accuracy,
            Sensibilidad,
            Especificidad,
            AUC,
            PuntajeGlobal
          ),
          ~ round(
            .x,
            4
          )
        )
      )

    datatable(
      tabla,
      rownames = FALSE,
      options = list(
        pageLength = 10,
        scrollX = TRUE
      )
    )
  })


  modelo_confusion_seleccionado <- reactive({

    req(
      analisis(),
      input$confusion_model
    )

    partes <- strsplit(
      input$confusion_model,
      ":::",
      fixed = TRUE
    )[[1]]

    validate(
      need(
        length(partes) == 2,
        "No fue posible identificar el modelo seleccionado."
      )
    )

    fila <- analisis()$resultados %>%
      filter(
        Modelo == partes[1],
        Grupo == partes[2]
      ) %>%
      slice(1)

    validate(
      need(
        nrow(fila) == 1,
        "No se encontró la matriz del modelo seleccionado."
      )
    )

    fila
  })


  output$confusion_table <- renderTable({

    fila <- modelo_confusion_seleccionado()

    total <- fila$VN + fila$FP + fila$FN + fila$VP

    porcentaje_vn <- 100 * fila$VN / total
    porcentaje_fp <- 100 * fila$FP / total
    porcentaje_fn <- 100 * fila$FN / total
    porcentaje_vp <- 100 * fila$VP / total

    data.frame(
      `Predicción / Real` = c(
        "Predicho negativo",
        "Predicho positivo"
      ),
      `Real negativo` = c(
        paste0("VN = ", round(porcentaje_vn, 1), "%"),
        paste0("FP = ", round(porcentaje_fp, 1), "%")
      ),
      `Real positivo` = c(
        paste0("FN = ", round(porcentaje_fn, 1), "%"),
        paste0("VP = ", round(porcentaje_vp, 1), "%")
      ),
      check.names = FALSE
    )
  },
  striped = TRUE,
  bordered = TRUE,
  spacing = "s"
  )


  output$winner_text <- renderUI({

    req(
      analisis()
    )

    mejor <- analisis()$mejor_modelo

    HTML(
      paste0(
        "<p><b>Modelo ganador:</b> ",
        mejor$Modelo,
        " con variables ",
        tolower(
          mejor$Grupo
        ),
        ".</p>",

        "<p><b>Accuracy:</b> ",
        round(
          mejor$Accuracy,
          4
        ),
        "</p>",

        "<p><b>Sensibilidad:</b> ",
        round(
          mejor$Sensibilidad,
          4
        ),
        "</p>",

        "<p><b>Especificidad:</b> ",
        round(
          mejor$Especificidad,
          4
        ),
        "</p>",

        "<p><b>AUC:</b> ",
        round(
          mejor$AUC,
          4
        ),
        "</p>",

        "<p><b>Puntaje global:</b> ",
        round(
          mejor$PuntajeGlobal,
          4
        ),
        ".</p>",

        "<p>El puntaje global es el promedio simple de las cuatro ",
        "métricas. Se utiliza como criterio inicial y pedagógico, ",
        "no como una regla clínica definitiva.</p>"
      )
    )
  })


  importancia_directa <- reactive({

    req(
      analisis(),
      input$importance_model_direct
    )

    bloque <- analisis()$resultado_directo

    fila_modelo <- analisis()$resultados %>%
      filter(
        Grupo == "Directo",
        Modelo == input$importance_model_direct
      ) %>%
      slice(1)

    validate(
      need(
        nrow(fila_modelo) == 1,
        "No se encontró el modelo directo seleccionado."
      )
    )

    modelo <- bloque$modelos[[
      input$importance_model_direct
    ]]

    calcular_importancia(
      modelo = modelo,
      algoritmo = input$importance_model_direct,
      preparacion = bloque$preparacion,
      auc_original = fila_modelo$AUC
    )
  })


  importancia_indirecta <- reactive({

    req(
      analisis(),
      input$importance_model_indirect
    )

    bloque <- analisis()$resultado_indirecto

    fila_modelo <- analisis()$resultados %>%
      filter(
        Grupo == "Indirecto",
        Modelo == input$importance_model_indirect
      ) %>%
      slice(1)

    validate(
      need(
        nrow(fila_modelo) == 1,
        "No se encontró el modelo indirecto seleccionado."
      )
    )

    modelo <- bloque$modelos[[
      input$importance_model_indirect
    ]]

    calcular_importancia(
      modelo = modelo,
      algoritmo = input$importance_model_indirect,
      preparacion = bloque$preparacion,
      auc_original = fila_modelo$AUC
    )
  })


  output$importance_plot_direct <- renderPlot({

    importancia <- importancia_directa() %>%
      slice_head(
        n = 12
      )

    ggplot(
      importancia,
      aes(
        x = reorder(
          Variable,
          Importancia
        ),
        y = Importancia
      )
    ) +
      geom_col() +
      coord_flip() +
      labs(
        x = NULL,
        y = "Caída del AUC",
        title = paste(
          "Directo:",
          input$importance_model_direct
        )
      ) +
      theme_minimal()
  })


  output$importance_plot_indirect <- renderPlot({

    importancia <- importancia_indirecta() %>%
      slice_head(
        n = 12
      )

    ggplot(
      importancia,
      aes(
        x = reorder(
          Variable,
          Importancia
        ),
        y = Importancia
      )
    ) +
      geom_col() +
      coord_flip() +
      labs(
        x = NULL,
        y = "Caída del AUC",
        title = paste(
          "Indirecto:",
          input$importance_model_indirect
        )
      ) +
      theme_minimal()
  })
}


# ============================================================
# 6. EJECUTAR EL DASHBOARD
# ============================================================

shinyApp(
  ui = ui,
  server = server
)
