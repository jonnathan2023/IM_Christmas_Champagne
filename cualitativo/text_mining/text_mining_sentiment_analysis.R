cat("\014")
rm(list = ls())
graphics.off()

# 1. Carga de paquetes ------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,   # Manipulación de datos (dplyr, stringr, tidyr, purrr)
  tidytext,    # Tokenización y minería de texto (TF, TF-IDF)
  tm,          # Stopwords base en español
  syuzhet,     # Léxico NRC de sentimientos y emociones en español
  widyr,       # Co-ocurrencias de atributos (Objetivo 2)
  naivebayes,  # Clasificador Naive Bayes rápido (bernoulli_naive_bayes)
  caret        # Folds estratificados y matriz de confusión
)

if (rstudioapi::isAvailable()) setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
options(pillar.sigfig = 3, dplyr.width = Inf)

# 2. Importación y parseo consolidado de los 7 archivos (.txt) ------
catalogo_archivos <- tibble(
  archivo = c(
    "FG1 20-29.txt",
    "FG2 (30-39).txt",
    "FG3 40-49.txt",
    "EP1(50 a 59).txt",
    "EP2(65-75).txt",
    "EP3(60-75).txt",
    "EP4(65-75 años).txt"
  ),
  sesion = c("FG1", "FG2", "FG3", "EP1", "EP2", "EP3", "EP4"),
  segmento = factor(
    c("20-29 años (FG1)", "30-39 años (FG2)", "40-49 años (FG3)",
      "50-59 años (EP1)", "60-75 años (EP2-EP4)", "60-75 años (EP2-EP4)", "60-75 años (EP2-EP4)"),
    levels = c("20-29 años (FG1)", "30-39 años (FG2)", "40-49 años (FG3)",
               "50-59 años (EP1)", "60-75 años (EP2-EP4)")
  )
)

leer_transcripcion <- function(archivo, sesion, segmento) {
  tibble(linea_texto = read_lines(archivo, locale = locale(encoding = "UTF-8"))) %>%
    mutate(
      linea_texto = str_remove(linea_texto, "^\\[\\d+:\\d+\\]\\s*"),
      linea_texto = str_squish(linea_texto)
    ) %>%
    filter(linea_texto != "", !str_detect(linea_texto, "^[2345]\\.\\s*(Bloque|Cierre)")) %>%
    extract(
      linea_texto,
      into = c("interlocutor", "texto"),
      regex = "^(?:([A-Za-zÁÉÍÓÚáéíóúñÑ\\s]+):\\s*)?(.*)$"
    ) %>%
    mutate(interlocutor = na_if(interlocutor, "")) %>%
    fill(interlocutor, .direction = "down") %>%
    mutate(
      sesion = sesion,
      segmento = segmento,
      interlocutor = case_when(
        sesion == "EP2" & interlocutor == "Entrevistada" ~ "Elena",
        interlocutor %in% c("Zaida", "Zayda") ~ "Zayda",
        interlocutor %in% c("Eli", "Elí") ~ "Elí",
        interlocutor %in% c("No identificado", "Participantes") ~ "Lucía", # Reasignar líneas grupales en FG1 para mantener n=5 reales
        TRUE ~ str_trim(interlocutor)
      )
    )
}

raw_corpus <- pmap_dfr(catalogo_archivos, leer_transcripcion) %>%
  mutate(id_intervencion = row_number())

# 3. Clasificación por Bloques (Objetivos Específicos) y filtrado ------
# Nota: Se evalúa de Cierre hacia atrás para que frases de transición como
# "Eso sería el segundo bloque. Pasemos al último bloque" (en EP3) clasifiquen en Obj3.
df_corpus <- raw_corpus %>%
  group_by(sesion) %>%
  mutate(
    es_moderador = interlocutor %in% c("Entrevistador", "Jonnathan Pedraza"),
    cambio_bloque = case_when(
      es_moderador & str_detect(texto, regex("varita mágica|diseñar el champán|diseñar el espumante", ignore_case = TRUE)) ~ "4_Cierre_Champan_Ideal",
      es_moderador & str_detect(texto, regex("dónde lo compran|hábitos de compra|en qué lugares lo comprarían|entrar ahora al bloque 3|último bloque|dónde realiza sus compras|dónde acostumbra comprar", ignore_case = TRUE)) ~ "Obj3_Habitos_Canales_Gasto",
      es_moderador & str_detect(texto, regex("frente a las opciones en tienda|frente a las opciones en la tienda|en el supermercado o buscando|entrar al bloque dos|cómo elige la bebida|relacionado a los factores|cómo elige el champán", ignore_case = TRUE)) ~ "Obj2_Atributos_Compra",
      es_moderador & str_detect(texto, regex("significa el momento del brindis|llega el momento de las doce|viaje mental|conversar sobre el brindis|primer bloque", ignore_case = TRUE)) ~ "Obj1_Simbolismo_Emociones",
      es_moderador & str_detect(texto, regex("rompiendo el hielo|romper el hielo|qué es lo que más disfrutan|cómo suele celebrar", ignore_case = TRUE)) ~ "0_Rompehielo",
      TRUE ~ NA_character_
    )
  ) %>%
  fill(cambio_bloque, .direction = "down") %>%
  ungroup() %>%
  mutate(objetivo = replace_na(cambio_bloque, "0_Rompehielo")) %>%
  filter(!es_moderador) %>%
  filter(!str_detect(texto, regex("Gracias por la invitación|Muchas gracias por hacernos|Quedo a disposición|Disponible para futuros|suerte con el trabajo|Espero que le vaya", ignore_case = TRUE)))

cat("\n====================================================================\n")
cat("1. RESUMEN DEL CORPUS CONSOLIDADO (7 TRANSCRIPCIONES - 17 PARTICIPANTES)\n")
cat("====================================================================\n")
df_corpus %>%
  group_by(sesion, segmento) %>%
  summarise(
    participantes    = n_distinct(interlocutor),
    intervenciones   = n(),
    palabras_totales = sum(str_count(texto, "\\S+")),
    .groups = "drop"
  ) %>%
  print(n = Inf)

# 4. Limpieza de texto y Tokenización (Unigramas) ------
stopwords_custom <- tibble(
  palabra = c(
    stopwords("es"),
    "si", "sí", "no", "pues", "eh", "hm", "ah", "así", "entonces", "bueno", "claro",
    "creo", "parece", "decir", "cómo", "digo", "va", "ser", "puede", "vez", "veces",
    "ok", "pucha", "acá", "allá", "ahí", "tal", "tipo", "tema", "cosas", "caso",
    "veo", "ver", "gusta", "gustan", "podría", "hacer", "hace", "hacemos", "van",
    "zaida", "zayda", "karin", "marianela", "rut", "eli", "elí", "tenorio",
    "teresa", "leonardo", "jonnathan", "lucía", "miquel", "vicky", "sebastián",
    "sebas", "tadeo", "juan", "alexandra", "ale", "carlos", "paola", "renzo",
    "claudia", "rocío", "elena", "dante", "osorio", "carmen", "señorita",
    "señora", "joven", "disculpa", "adelante", "gracias", "buenas", "noches",
    "tardes", "hola", "pregunta", "respuesta", "año", "años", "siempre", "bastante"
  )
)

tokens_corpus <- df_corpus %>%
  mutate(
    texto = str_replace_all(texto, regex("plaza\\s+bea|plaza\\s+vea", ignore_case = TRUE), "plazavea"),
    texto = str_replace_all(texto, regex("mesa\\s+redonda", ignore_case = TRUE), "mesaredonda"),
    texto = str_replace_all(texto, regex("semi-seco|semi\\s+seco", ignore_case = TRUE), "semiseco"),
    texto = str_replace_all(texto, regex("semi-dulces|semi\\s+dulce", ignore_case = TRUE), "semidulce")
  ) %>%
  unnest_tokens(input = texto, output = palabra, token = "words") %>%
  filter(!str_detect(palabra, "^\\d+$")) %>%
  anti_join(stopwords_custom, by = "palabra") %>%
  mutate(
    palabra = case_when(
      palabra %in% c("ricadona", "riccadonna") ~ "riccadonna",
      palabra %in% c("quierolo", "queirolo") ~ "queirolo",
      palabra %in% c("champanes", "champan", "champán", "champaña") ~ "champán",
      palabra %in% c("espumantes", "espumante", "espumoso", "espumosas") ~ "espumante",
      palabra %in% c("dulcecito", "dulcito", "dulces", "dulce") ~ "dulce",
      palabra %in% c("burbujitas", "burbujeante", "burbujas") ~ "burbujas",
      palabra %in% c("rosa", "rosé", "rose") ~ "rosé",
      palabra %in% c("abuelito", "abuelo", "abuelos") ~ "abuelo",
      palabra %in% c("botellita", "botellas", "botella") ~ "botella",
      palabra %in% c("supermercados", "supermercado") ~ "supermercado",
      palabra %in% c("licorerías", "licorería") ~ "licorería",
      palabra %in% c("vinito", "vino") ~ "vino",
      palabra %in% c("chela", "cerveza") ~ "cerveza",
      TRUE ~ palabra
    )
  )

# 5. Análisis de Frecuencias (TF) y Términos Característicos (TF-IDF) ------
cat("\n====================================================================\n")
cat("2. TOP 8 TÉRMINOS MÁS FRECUENTES (TF) POR OBJETIVO ESPECÍFICO\n")
cat("====================================================================\n")
tokens_corpus %>%
  filter(str_starts(objetivo, "Obj")) %>%
  count(objetivo, palabra, sort = TRUE) %>%
  group_by(objetivo) %>%
  slice_max(n, n = 8, with_ties = FALSE) %>%
  summarise(top_terminos = paste0(palabra, " (", n, ")", collapse = ", "), .groups = "drop") %>%
  print(n = Inf)

cat("\n====================================================================\n")
cat("3. TÉRMINOS MÁS DISTINTIVOS POR SEGMENTO DE EDAD (TF-IDF)\n")
cat("====================================================================\n")
tokens_corpus %>%
  filter(!palabra %in% c("champán", "espumante", "botella")) %>%
  count(segmento, palabra, sort = TRUE) %>%
  bind_tf_idf(palabra, segmento, n) %>%
  filter(n >= 2) %>%
  group_by(segmento) %>%
  slice_max(tf_idf, n = 6, with_ties = FALSE) %>%
  summarise(
    terminos_distintivos = paste0(palabra, " (n=", n, ", tf_idf=", round(tf_idf, 3), ")", collapse = " | "),
    .groups = "drop"
  ) %>%
  print(n = Inf)

# 6. Análisis de Sentimiento y Emociones (Léxico NRC) ------
nrc_scores <- get_nrc_sentiment(tokens_corpus$palabra, language = "spanish")
tokens_nrc <- bind_cols(tokens_corpus, nrc_scores)

cat("\n====================================================================\n")
cat("4. PERFIL EMOCIONAL ASOCIADO AL BRINDIS (OBJETIVO 1 - LÉXICO NRC)\n")
cat("====================================================================\n")
tokens_nrc %>%
  filter(objetivo %in% c("0_Rompehielo", "Obj1_Simbolismo_Emociones")) %>%
  summarise(
    Alegria            = sum(joy),
    Confianza          = sum(trust),
    Anticipacion       = sum(anticipation),
    Sorpresa           = sum(surprise),
    Tristeza_Nostalgia = sum(sadness),
    Miedo              = sum(fear),
    Ira                = sum(anger),
    Disgusto           = sum(disgust)
  ) %>%
  pivot_longer(everything(), names_to = "Dimension_Emocional", values_to = "Frecuencia") %>%
  mutate(Porcentaje = round(100 * Frecuencia / sum(Frecuencia), 1)) %>%
  arrange(desc(Frecuencia)) %>%
  print(n = Inf)

cat("\n--- Distribución de Polaridad y Emociones clave por Segmento Etario (%) ---\n")
tokens_nrc %>%
  group_by(segmento) %>%
  summarise(
    Tokens_Evaluados   = sum(positive + negative > 0),
    Pct_Positivo       = round(100 * sum(positive) / sum(positive + negative), 1),
    Pct_Negativo       = round(100 * sum(negative) / sum(positive + negative), 1),
    Alegria_Count      = sum(joy),
    Confianza_Count    = sum(trust),
    Tristeza_Count     = sum(sadness),
    Miedo_Riesgo_Count = sum(fear),
    .groups = "drop"
  ) %>%
  print(n = Inf)

# 7. Clasificador Naive Bayes con Priors Balanceados (5-Fold CV + Laplace = 1) ------
cat("\n====================================================================\n")
cat("5. CLASIFICADOR NAIVE BAYES DE SENTIMIENTO (5-FOLD CV + PRIORS BALANCEADOS)\n")
cat("====================================================================\n")
terminos_riesgo_cautela <- c(
  "seco", "seca", "secos", "amargo", "agrio", "ácido", "horrible", "decepcionar",
  "adulteración", "adulterados", "imitados", "intoxicaciones", "tristeza", "tristes",
  "falleció", "difíciles", "colas", "caro", "arriesgaría", "malogra", "fea", "nostalgia"
)

intervenciones_etiquetadas <- tokens_nrc %>%
  mutate(es_riesgo = as.integer(palabra %in% terminos_riesgo_cautela)) %>%
  group_by(id_intervencion, sesion, segmento, interlocutor, objetivo) %>%
  summarise(
    score_pos = sum(positive),
    score_neg = sum(negative) + 2 * sum(es_riesgo),
    n_tokens  = n(),
    .groups = "drop"
  ) %>%
  mutate(
    clase_sentimiento = factor(
      ifelse(score_neg >= 2 & score_neg >= score_pos * 0.5, "Cautela_Riesgo_Nostalgia", "Positivo_Festivo"),
      levels = c("Positivo_Festivo", "Cautela_Riesgo_Nostalgia")
    )
  )

vocabulario_nb <- tokens_corpus %>%
  count(palabra) %>%
  filter(n >= 2) %>%
  pull(palabra)

dtm_nb <- tokens_corpus %>%
  filter(palabra %in% vocabulario_nb) %>%
  distinct(id_intervencion, palabra) %>%
  mutate(presencia = 1L) %>%
  pivot_wider(
    names_from = palabra,
    values_from = presencia,
    values_fill = 0L
  )

datos_nb <- intervenciones_etiquetadas %>%
  inner_join(dtm_nb, by = "id_intervencion")

x_matriz <- as.matrix(datos_nb %>% select(all_of(vocabulario_nb)))
y_vector <- datos_nb$clase_sentimiento

# Validación Cruzada Estratificada 5-Fold con Priors Uniformes c(0.5, 0.5) para corregir desbalance
set.seed(2026)
folds <- createFolds(y_vector, k = 5, list = TRUE, returnTrain = FALSE)
predicciones_cv5 <- rep(NA_character_, length(y_vector))

for (idx_test in folds) {
  modelo_fold <- bernoulli_naive_bayes(
    x = x_matriz[-idx_test, , drop = FALSE],
    y = y_vector[-idx_test],
    prior = c(0.5, 0.5),
    laplace = 1
  )
  probs_fold <- predict(modelo_fold, newdata = x_matriz[idx_test, , drop = FALSE], type = "prob")
  predicciones_cv5[idx_test] <- ifelse(
    probs_fold[, "Cautela_Riesgo_Nostalgia"] >= 0.45,
    "Cautela_Riesgo_Nostalgia",
    "Positivo_Festivo"
  )
}
predicciones_cv5 <- factor(predicciones_cv5, levels = levels(y_vector))

cm_nb <- confusionMatrix(predicciones_cv5, y_vector, positive = "Cautela_Riesgo_Nostalgia")

cat("\n--- Métricas Globales del Modelo Naive Bayes (5-Fold CV, Laplace = 1) ---\n")
tibble(
  Metrica = c("Accuracy (Exactitud)", "Balanced Accuracy", "Sensitivity (Detección Cautela/Riesgo)", "Specificity (Detección Positivo/Festivo)", "Kappa"),
  Valor   = c(
    cm_nb$overall["Accuracy"],
    cm_nb$byClass["Balanced Accuracy"],
    cm_nb$byClass["Sensitivity"],
    cm_nb$byClass["Specificity"],
    cm_nb$overall["Kappa"]
  )
) %>%
  mutate(Valor = round(Valor, 4)) %>%
  print()

cat("\n--- Matriz de Confusión (5-Fold CV) ---\n")
print(cm_nb$table)

# Modelo final con priors balanceados
modelo_nb_final <- bernoulli_naive_bayes(x = x_matriz, y = y_vector, prior = c(0.5, 0.5), laplace = 1)
probs_posteriores <- predict(modelo_nb_final, newdata = x_matriz, type = "prob")

cat("\n--- Resumen de Predicciones Naive Bayes por Objetivo Específico ---\n")
datos_nb %>%
  mutate(
    pred_nb = ifelse(probs_posteriores[, "Cautela_Riesgo_Nostalgia"] >= 0.45, "Cautela_Riesgo_Nostalgia", "Positivo_Festivo"),
    prob_positivo = probs_posteriores[, "Positivo_Festivo"],
    prob_cautela  = probs_posteriores[, "Cautela_Riesgo_Nostalgia"]
  ) %>%
  group_by(objetivo) %>%
  summarise(
    Intervenciones        = n(),
    Pred_Positivo_Festivo = sum(pred_nb == "Positivo_Festivo"),
    Pred_Cautela_Riesgo   = sum(pred_nb == "Cautela_Riesgo_Nostalgia"),
    Prob_Media_Positivo   = round(mean(prob_positivo), 3),
    Prob_Media_Cautela    = round(mean(prob_cautela), 3),
    .groups = "drop"
  ) %>%
  print(n = Inf)

cat("\n--- Top Palabras Discriminantes por Clase en el Modelo Naive Bayes ---\n")
tabla_probs <- coef(modelo_nb_final)
razon_verosimilitud <- tibble(
  palabra          = vocabulario_nb,
  prob_en_Positivo = tabla_probs[, "Positivo_Festivo:1"],
  prob_en_Cautela  = tabla_probs[, "Cautela_Riesgo_Nostalgia:1"],
  log_odds_ratio   = log(prob_en_Positivo / prob_en_Cautela)
)

cat("A) Términos con mayor peso hacia 'Positivo_Festivo':\n")
razon_verosimilitud %>%
  arrange(desc(log_odds_ratio)) %>%
  slice_head(n = 8) %>%
  print()

cat("\nB) Términos con mayor peso hacia 'Cautela_Riesgo_Nostalgia' (Aversión al riesgo y duelo):\n")
razon_verosimilitud %>%
  arrange(log_odds_ratio) %>%
  slice_head(n = 8) %>%
  print()

# 8. Co-ocurrencia de Atributos Limpia - Objetivo 2 (Consola) ------
cat("\n====================================================================\n")
cat("6. TOP 15 PARES DE CO-OCURRENCIA DE ATRIBUTOS (OBJETIVO 2 + CIERRE)\n")
cat("====================================================================\n")
tokens_corpus %>%
  filter(objetivo %in% c("Obj2_Atributos_Compra", "4_Cierre_Champan_Ideal")) %>%
  filter(!palabra %in% c("champán", "espumante", "botella", "comprar", " compraría", "alguien", "opciones")) %>%
  pairwise_count(palabra, id_intervencion, sort = TRUE, upper = FALSE) %>%
  slice_head(n = 15) %>%
  print(n = 15)

# 9. Extracción Cuantitativa de Presupuesto, Canales y Sustitutos - Objetivo 3 ------
cat("\n====================================================================\n")
cat("7. EXTRACCIÓN DE PRESUPUESTO EN SOLES POR SEGMENTO (OBJETIVO 3)\n")
cat("====================================================================\n")
presupuesto_consolidado <- df_corpus %>%
  filter(objetivo == "Obj3_Habitos_Canales_Gasto" | str_detect(texto, "60, 70 soles")) %>%
  mutate(
    texto_num = str_replace_all(texto, regex("cuarenta y cinco", ignore_case = TRUE), "45"),
    texto_num = str_replace_all(texto, regex("cuarenta", ignore_case = TRUE), "40"),
    montos = str_extract_all(texto_num, "\\b\\d{2,3}\\b")
  ) %>%
  unnest(montos) %>%
  mutate(monto_soles = as.numeric(montos)) %>%
  filter(monto_soles %in% c(11, 16, 18, 22, 25, 35, 40, 45, 50, 60, 70, 80, 100, 120, 130)) %>%
  group_by(segmento) %>%
  summarise(
    participantes_con_dato = n_distinct(interlocutor),
    min_soles          = min(monto_soles),
    mediana_soles      = median(monto_soles),
    promedio_soles     = round(mean(monto_soles), 1),
    max_soles          = max(monto_soles),
    valores_declarados = paste(sort(unique(monto_soles)), collapse = ", "),
    .groups = "drop"
  )

print(presupuesto_consolidado, n = Inf)

cat("\n====================================================================\n")
cat("8. FRECUENCIA DE CANALES Y SUSTITUTOS POR SEGMENTO (OBJETIVO 3)\n")
cat("====================================================================\n")
diccionario_obj3 <- tibble(
  palabra = c("supermercado", "plazavea", "metro", "tottus", "licorería", "mesaredonda", "bodega", "internet",
              "vino", "sidra", "cerveza", "ron"),
  tipo = c(rep("1_Canal", 8), rep("2_Sustituto", 4)),
  etiqueta = c("Supermercado (Genérico)", "Plaza Vea", "Metro", "Tottus", "Licorería", "Mesa Redonda (Mayorista)", "Bodega (Urgencia)", "Internet (Mención de rechazo)",
               "Vino (Semiseco/Dulce/Rosé)", "Sidra", "Cerveza", "Ron")
)

tokens_corpus %>%
  filter(objetivo == "Obj3_Habitos_Canales_Gasto" | palabra == "sidra") %>%
  inner_join(diccionario_obj3, by = "palabra") %>%
  count(tipo, etiqueta, segmento) %>%
  pivot_wider(names_from = segmento, values_from = n, values_fill = 0) %>%
  mutate(Total_Menciones = rowSums(across(where(is.numeric)))) %>%
  arrange(tipo, desc(Total_Menciones)) %>%
  print(n = Inf)

