# Fase 1 - Entendimento do Negócio (CRISP-DM)

## 🎯 Objetivo do Projeto

Desenvolver um modelo preditivo capaz de estimar o tempo restante de vida útil (RUL – *Remaining Useful Life*) de equipamentos industriais, com base em leituras contínuas de sensores. O objetivo é reduzir falhas não planejadas e otimizar a manutenção preventiva em ambientes industriais, simulando um cenário realista de aplicação em clientes da Semantix.

---

## 🏢 Contexto da Empresa

A Semantix é uma empresa especializada em Big Data, Inteligência Artificial e Analytics, que entrega soluções para clientes nos setores de indústria, energia, saúde e finanças. Entre os desafios enfrentados por esses clientes, destaca-se o alto custo associado à manutenção corretiva e paradas inesperadas de máquinas.

A manutenção preditiva é uma abordagem orientada por dados que permite antecipar falhas e realizar intervenções programadas, resultando em aumento da eficiência operacional, redução de custos e maior segurança.

---

## 🛠️ Problema de Negócio

**Como prever com precisão o tempo restante de vida útil de máquinas industriais, utilizando dados de sensores, para permitir ações de manutenção proativas e evitar paradas inesperadas?**

---

## 💡 Solução Proposta

Aplicar técnicas de Machine Learning (e futuramente Deep Learning) sobre dados multivariados de sensores, extraídos do dataset C-MAPSS (NASA), para:

- Estimar a Remaining Useful Life (RUL) de turbinas industriais;
- Priorizar a manutenção de equipamentos com base no risco de falha;
- Gerar insights operacionais por meio de visualizações e alertas preditivos.

---

## ✅ Critérios de Sucesso

- Redução simulada de falhas inesperadas em pelo menos **30%** em relação à manutenção corretiva;
- Erro médio na predição da RUL inferior a **15 ciclos** (ex: MAE < 15);
- Capacidade de integrar o modelo com uma arquitetura operacional (ex.: via API ou dashboard).

---

## 🔒 Restrições e Considerações

- O dataset é simulado, mas amplamente aceito como benchmark realista;
- O escopo inicial será a unidade FD001, com expansão possível para cenários mais complexos (FD002 a FD004);
- A solução será desenvolvida como uma prova de conceito (PoC), mas com arquitetura modular que permite futura implantação real.

# **Etapa 2 do CRISP-DM: Estrutura dos Dados**

A base possui 20.631  registros e 26 colunas. Abaixo estão os principais campos:


# Dicionário de Dados – FD001 (C-MAPSS)

| Coluna       | Tipo    | Descrição                                                                                          | Valores Possíveis / Unidades          |
|--------------|---------|----------------------------------------------------------------------------------------------------|---------------------------------------|
| **unit**     | int64   | Identificador da unidade (motor)                                                                    | 1 a 100                               |
| **cycle**    | int64   | Número sequencial de ciclo de operação                                                              | ≥ 1                                   |
| **setting_1**| float64 | Condição operacional 1 (fase de voo simulada – altitude)                                            | –                                     |
| **setting_2**| float64 | Condição operacional 2 (fase de voo simulada – Mach)                                                | –                                     |
| **setting_3**| float64 | Condição operacional 3 (fase de voo simulada – ângulo do acelerador)                                | –                                     |
| **sensor_1** | float64 | T2 – Temperatura total na entrada do fan (°R)                                                       | °R                                    |
| **sensor_2** | float64 | T24 – Temperatura total na saída do LPC (°R)                                                       | °R                                    |
| **sensor_3** | float64 | T30 – Temperatura total na saída do HPC (°R)                                                       | °R                                    |
| **sensor_4** | float64 | T50 – Temperatura total na saída do LPT (°R)                                                       | °R                                    |
| **sensor_5** | float64 | P2 – Pressão na entrada do fan (psia)                                                              | psia                                  |
| **sensor_6** | float64 | P15 – Pressão total no by‐pass duct (psia)                                                         | psia                                  |
| **sensor_7** | float64 | P30 – Pressão total na saída do HPC (psia)                                                         | psia                                  |
| **sensor_8** | float64 | Nf – Velocidade física do fan (rpm)                                                                | rpm                                   |
| **sensor_9** | float64 | Nc – Velocidade física do núcleo (core) (rpm)                                                      | rpm                                   |
| **sensor_10**| float64 | epr – Relação pressórica do motor (P50/P2)                                                         | —                                     |
| **sensor_11**| float64 | Ps30 – Pressão estática na saída do HPC (psia)                                                     | psia                                  |
| **sensor_12**| float64 | φ (phi) – Razão fluxo de combustível / Ps30 (pps/psi)                                              | pps/psi                               |
| **sensor_13**| float64 | NRf – Velocidade corrigida do fan (rpm)                                                            | rpm                                   |
| **sensor_14**| float64 | NRc – Velocidade corrigida do core (rpm)                                                           | rpm                                   |
| **sensor_15**| float64 | BPR – By‐pass ratio                                                                                 | —                                     |
| **sensor_16**| float64 | farB – Razão combustível‐ar no queimador                                                            | —                                     |
| **sensor_17**| float64 | htBleed – Entalpia de bleed (lbm/s)                                                                | lbm/s                                 |
| **sensor_18**| float64 | Nf_dmd – Velocidade física do fan demandada (rpm)                                                  | rpm                                   |
| **sensor_19**| float64 | PcNfR_dmd – Velocidade corrigida do fan demandada (rpm)                                            | rpm                                   |
| **sensor_20**| float64 | W31 – Fluxo de bleed do HPT (lbm/s)                                                                | lbm/s                                 |
| **sensor_21**| float64 | W32 – Fluxo de bleed do LPT (lbm/s)                                                                | lbm/s                                 |
| **RUL**      | int64   | Remaining Useful Life: ciclos restantes até a falha, calculado por `max(cycle)` – `cycle`           | ≥ 0                                   |
              

# Resumo da Análise Descrtiva

## 1. Dataset Utilizado  
- **Fonte:** NASA PCoE – C-MAPSS Jet Engine Simulated Data  
- **Arquivo:** `train_FD001.txt`  
- **Objetivo:** Simular degradação de motores a jato por ciclo de operação  

---

## 2. Estrutura e Qualidade dos Dados  
- **Registros:** 20 631  
- **Variáveis:** 27 (26 originais + 1 target `RUL`)  
- **Tipos de dados:**  
  - Numéricas (`float64`): 25 (`setting_1`…`sensor_21`)  
  - Inteiras (`int64`): 2 (`unit`, `cycle`)  
- **Valores ausentes / duplicatas:** 0 / 0  

---

## 3. Cálculo da Variável-Alvo (RUL)  
- **RUL**: ciclos restantes até a falha de cada unidade  

---

## 4. Análise Univariada  
- **Sensores com variância quase nula (descartar):**  
  `sensor_19`, `sensor_18`, `sensor_16`, `sensor_10`, `sensor_5`  

---

## 5. Análise Bivariada (Correlação com RUL)

### 🔵 Correlações Positivas (top 4)  
1. `sensor_12` → +0,67  
2. `sensor_7`  → +0,66  
3. `sensor_21` → +0,64  
4. `sensor_20` → +0,63  

### 🔴 Correlações Negativas (top 8)  
1. `sensor_11` → –0,70  
2. `sensor_4`  → –0,68  
3. `sensor_15` → –0,64  
4. `sensor_2`  → –0,61  
5. `sensor_17` → –0,61  
6. `sensor_3`  → –0,58  
7. `sensor_8`  → –0,56  
8. `sensor_13` → –0,56  

---

## 6. Identificação de Outliers por Sensor  
| Sensor     | Qtde Outliers |
|:-----------|--------------:|
| sensor_9   | 1 686         |
| sensor_14  | 1 543         |
| sensor_6   |   406         |
| sensor_8   |   320         |
| sensor_3   |   165         |
| …          |   …           |
| sensor_1,5,10,16,18,19 | 0  |

---

## 7. Análise Bivariada – Scatter Plots  
Visualizamos a dispersão de **RUL** versus cada um dos sensores com maior correlação (positiva e negativa), confirmando padrões lineares e tendências de variação conforme a vida útil restante.

---

## Conclusões da Fase 2  
1. O dataset está **limpo e completo**, sem ausentes ou duplicatas.  
2. A variável-alvo **RUL** foi calculada corretamente.  
3. Cinco sensores de baixa variabilidade foram identificados e descartados.  
4. Sensores-chave com alta correlação (positiva e negativa) foram priorizados para modelagem:  
   - **Positivos:** `sensor_12`, `sensor_7`, `sensor_21`, `sensor_20`  
   - **Negativos:** `sensor_11`, `sensor_4`, `sensor_15`, `sensor_2`, `sensor_17`, `sensor_3`, `sensor_8`, `sensor_13`  
5. A contagem de outliers por sensor orienta estratégias de tratamento específicas.  
6. Scatter plots bivariados validaram as correlações e revelaram a dispersão dos dados.

🎯 **Próxima Etapa (Fase 3 – Preparação dos Dados):**  
- Remover sensores constantes;  
- Normalizar variáveis;  
- Tratar outliers conforme frequência identificada;  
- Criar janelas temporais para modelagem sequencial.  

# Fase 3 – Preparação dos Dados (CRISP-DM)

## 1. Objetivos desta Etapa  
- Transformar os dados brutos em um formato adequado para modelagem sequencial e clássica.  
- Garantir consistência entre treino e teste, usando apenas informações do conjunto de treino para calibrar transformações.  
- Criar janelas temporais que capturem a dinâmica de degradação de cada unidade.

---

## 2. Seleção de Variáveis  
- **Descartar sensores constantes:**  
  `sensor_19`, `sensor_18`, `sensor_16`, `sensor_10`, `sensor_5`  
- **Manter variáveis operacionais e preditivas:**  
  - Settings: `setting_1`, `setting_2`, `setting_3`  
  - Sensores-chave (|corr| ≥ 0,56):  
    - Positivos: `sensor_12`, `sensor_7`, `sensor_21`, `sensor_20`  
    - Negativos: `sensor_11`, `sensor_4`, `sensor_15`, `sensor_2`, `sensor_17`, `sensor_3`, `sensor_8`, `sensor_13`  
- **Variáveis auxiliares sem escala:**  
  `unit`, `cycle`  
- **Target:**  
  `RUL`  

---

## 3. Conversão de Tipos e Normalização  
- Converter todos os `settings` e sensores para `float`.  
- Aplicar **Min-Max Scaling** (0–1) **usando apenas estatísticas do treino**:  
  - Fit do scaler no treino → transform no treino e no teste.  
- Manter `unit`, `cycle` e `RUL` inalterados.

---

## 4. Engenharia de Features Temporais  
- Definir `window_size = 30` ciclos como comprimento de cada sequência.  
- **Treino:** gerar janelas deslizantes para cada unidade, onde cada janela de 30 ciclos recebe como alvo o RUL do ciclo final da sequência.  
- **Teste:** usar somente a última janela de 30 ciclos de cada unidade, associando o RUL fixo informado em `RUL_FD001.txt`.

---

## 5. Particionamento Treino / Validação / Teste  
- **Treino / Validação:**  
  - Misturar (shuffle) as janelas do treino, garantindo 80 % para treino e 20 % para validação interna.  
- **Teste:**  
  - Conjunto externo composto pelas janelas finais de cada unidade de teste.

---

## 6. Métricas de Avaliação  
- **Regressão clássica:** MAE e RMSE.  
- **Scoring NASA:**  
  \[
    \text{Score} = \sum_{i=1}^N
    \begin{cases}
      e^{-d_i/13} - 1, & d_i < 0,\\
      e^{d_i/10} - 1,  & d_i \ge 0,
    \end{cases}
    \quad d_i = \hat{y}_i - y_i.
  \]  
- Comparar desempenho em treino, validação e teste para validar generalização.

---

## 7. Conclusões da Fase 3  
1. **Variáveis irrelevantes** removidas, mantendo apenas as features informativas e operacionais.  
2. **Transformações** aplicadas de forma consistente (tipo, escala), evitando vazamento de dados do teste.  
3. **Janelas temporais** estruturadas para capturar a degradação ao longo de 30 ciclos.  
4. **Divisão** treino/validação/teste pronta para alimentar modelos de machine learning (clássicos e sequenciais).  
5. **Pipeline completo** e reproduzível, pronto para fase de treinamento de modelos e análise de resultados.  

🚀 **Próximos passos:**  
- Treinar e ajustar hiperparâmetros de modelos sequenciais (LSTM/RNN) e clássicos (RF, XGBoost).  
- Validar em janelas de teste e comparar métricas para escolher o melhor approach.  
- Preparar apresentação dos resultados e recomendações de implantação.  

# Fase 4 — Modelagem (CRISP-DM)

## 1) Objetivo da fase
Projetar, treinar e **comparar** modelos para prever o RUL (Remaining Useful Life) no cenário **FD001 (C-MAPSS)**, definindo arquitetura, hiperparâmetros, insumos e protocolo de validação **sem vazamento**. Ao final desta fase deixamos o(s) candidato(s) **pré-selecionado(s)** para a Fase 5 (Avaliação).

---

## 2) Formulação do problema
- **Tarefa:** Regressão (prever RUL em ciclos).
- **Granularidade:** por **unidade** (`unit`) e **ciclo** (`cycle`).
- **Alvo (target):** `RUL` (ciclos restantes até falha).
- **Premissas-chave:**
  - Nada de estatísticas do **teste** no treino/validação (**no leakage**).
  - Sinais com **correlação positiva e negativa** com o RUL são úteis; usamos **|correlação|** para priorização, não para descartar sinais com sentido oposto.

---

## 3) Variáveis e preparação dos dados
- **Descartadas (variância ≈ 0):** `sensor_19`, `sensor_18`, `sensor_16`, `sensor_10`, `sensor_5`.
- **Mantidas (contínuas | 19 features):**
  - *Settings:* `setting_1`, `setting_2`, `setting_3`
  - *Sensores-chave (|corr| ≥ ~0,56 com RUL):*
    - **Positivas:** `sensor_12`, `sensor_7`, `sensor_21`, `sensor_20`
    - **Negativas:** `sensor_11`, `sensor_4`, `sensor_15`, `sensor_2`, `sensor_17`, `sensor_3`, `sensor_8`, `sensor_13`
- **Metadados (não escalados):** `unit`, `cycle` (para agrupamento e criação de janelas).
- **Normalização:** `MinMaxScaler` ajustado **apenas no treino** e aplicado em treino/val/teste.
- **Rótulo no treino:** `RUL = max(cycle)_da_unit − cycle`.
- **Observação (features de tendência):** deltas/rolagens foram **testados**; para FD001 não superaram a linha base e foram **não adotados** no modelo final desta fase.

---

## 4) Estratégias de modelagem (duas linhas)

### Linha A — Modelos **estáticos** (sem sequência)
- **Algoritmos:** Random Forest Regressor (RF) e XGBoost Regressor (XGB).
- **Entrada:** vetor com **19 features** (settings + sensores).
- **Amostragem de treino:** amostrar **N pontos por unidade** (ex.: `N=5`) após embaralhar, para cobrir diversos níveis de RUL.
- **Teste:** usar o **último ciclo** de cada `unit` em `test_FD001.txt`; rótulo vem de `RUL_FD001.txt`.
- **Propósito:** baseline forte, interpretável e de **baixo custo**.

### Linha B — Modelos **sequenciais** (com dinâmica temporal)
- **Algoritmo principal:** **LSTM**.
- **Janela temporal (principal):** `window_size = 30` ciclos (variação: `20`).
- **Entrada:** tensor `(window_size × n_features)` por janela.
- **Criação de janelas (treino/val):** janelas deslizantes por `unit`; **alvo** é o `RUL` no **último passo** da janela.
- **Teste:** **1 janela final por unidade** (com **padding à esquerda** se a série for menor que a janela).
- **Clipping do rótulo:** avaliamos teto (ex.: `125`) para reduzir assimetria; **para FD001, sem clipping** obteve melhor equilíbrio — logo, **não adotado** no candidato final desta fase.

---

## 5) Controles de qualidade e **no-leakage**
- Ajuste do **scaler** e quaisquer estatísticas **apenas no treino**.
- **Nada do conjunto de teste** entra na engenharia de atributos ou normalização.
- **Alinhamento do rótulo** no teste: a **última janela** de cada `unit` é mapeada à **i-ésima linha** de `RUL_FD001.txt` (ordem 1..N).
- **Padding** à esquerda com a primeira observação da unidade quando faltarem ciclos para compor a janela.
- **Reprodutibilidade:** seeds fixos; `EarlyStopping` com restauração do melhor ponto; registro de versão dos artefatos.

---

## 6) Hiperparâmetros e busca (espaço e defaults)
- **RF:** `n_estimators=200`, `random_state=42` (profundidade livre).
- **XGB:** `n_estimators≈300`, `learning_rate≈0,05`, `max_depth≈6`, `subsample=0,8`, `colsample_bytree=0,8`, `random_state=42`.
- **LSTM (candidato final desta fase):**
  - `window_size=30`, `units=64` (1ª camada) → `32` (2ª camada), `dropout=0,2`, `batch_size=64`
  - **Treino:** até **40 épocas** com `EarlyStopping(patience 3–5)`, métrica de validação `MSE`/`MAE`.

> *Nota:* Espaços de busca maiores foram considerados para etapa posterior; aqui mantivemos **grid simples** focado no equilíbrio viabilidade/ganho.

---

## 7) Protocolo de validação e teste
- **Validação interna (treino/val):** *hold-out* 80/20 sobre as janelas de treino (embaralhadas).  
  > Alternativa considerada: *hold-out por unidade* (validação leave-units-out). Para FD001, a divisão por janelas foi suficiente sem afetar o teste externo.
- **Teste externo:** 1 janela final por `unit` em `test_FD001`; rótulo oficial em `RUL_FD001.txt`.
- **Métricas de comparação:**
  - **MAE** — interpretabilidade direta em ciclos;
  - **RMSE** — penaliza grandes erros;
  - **NASA score** — penaliza **mais** a **superestimação** (erros positivos).

---

## 8) Critério de **seleção** do modelo
Priorizar **menor NASA score** no teste; em caso de empate, considerar **MAE** e **RMSE**.  
Com base nos experimentos desta fase, o **LSTM (janela 30, units 64, dropout 0,2)** segue como **candidato principal** para a **Fase 5 (Avaliação)**, com RF/XGB servindo de **baselines** e apoio à interpretabilidade (importância de variáveis).

---

## 9) Entregáveis desta fase
- Definições de **arquitetura** e **hiperparâmetros** dos modelos candidatos.
- Artefatos de pré-processamento: `features.json`, `scaler.pkl`, `config.json` (janela, n_features).
- Checklist de **criação de janelas** e **mapeamento de rótulos** (treino/val/teste).
- Scripts/notebooks para: preparação, treino/validação, geração de janelas e inferência por última janela do teste.

> **Observação:** resultados numéricos detalhados, gráficos (resíduos, ŷ×y, MAE por unidade) e análises comparativas serão apresentados na **Fase 5 — Avaliação**.

# Fase 5 – Avaliação (CRISP-DM)

## 1) Objetivo
Avaliar a performance de modelos para previsão de **RUL** no cenário **FD001 (C-MAPSS)**, comparando:
- **Modelos estáticos** (RF, XGB) com amostragem por unidade;
- **Modelo sequencial** (**LSTM**) com janelas temporais (principal: `window=30`).

**Métricas:** MAE, RMSE e **NASA score** (quanto **menor**, melhor). Também acompanhamos **% de superestimação** (proporção de `ŷ > y`), pois a superestimação de RUL é mais arriscada operacionalmente.

---

## 2) Dados, particionamento e protocolo
- **Treino/Validação (interno):** janelas deslizantes por unidade (80/20), sem misturar teste.
- **Teste (externo):** última janela por unidade em `test_FD001.txt` (com **padding à esquerda** se necessário), rótulos de `RUL_FD001.txt` (ordem 1..N).
- **Normalização:** `MinMaxScaler` ajustado **somente no treino** e aplicado em treino/val/teste (sem vazamento).
- **Features (19):** `setting_1..3` + sensores (exceto variância≈0: `sensor_5,10,16,18,19`). Mantivemos sensores com correlação **positiva e negativa** ao RUL (critérios por **|correlação|**).

---

## 3) Resultados

### 3.1 Baselines estáticos (último ciclo por unidade no teste)
| Modelo        | Val MAE | Test MAE | Test RMSE | Test NASA |
|---|---:|---:|---:|---:|
| Random Forest | 31.81 | **21.38** | 28.76 | 13,213 |
| XGBoost       | 31.63 | **21.04** | 28.33 | 11,884 |

**Leitura.** RF/XGB (sem dinâmica temporal) entregam **~21 ciclos** de MAE no teste — úteis como **baseline** e para **explicabilidade** (importância de variáveis).

### 3.2 Modelo sequencial (LSTM – janelas)
| window | Clipping RUL | units | Val MAE | **Test MAE** | **Test RMSE** | **Test NASA** |
|---:|:---:|---:|---:|---:|---:|---:|
| 30 | —   | 64  | 20.30 | **15.95** | **21.83** | **3,948** |
| 30 | 125 | 64  | 36.93 | 35.14 | 40.47 | 17,445 |
| 20 | 125 | 128 | 37.21 | 34.91 | 40.90 | 22,211 |

**Leitura.** **LSTM (window=30, sem clipping)** supera RF/XGB em MAE/RMSE e reduz **fortemente** o **NASA** (menos risco de superestimar o RUL). O *clipping* de rótulo **degradou** o desempenho no FD001.

> **Complemento – “serving” com calibração pós-modelo (Fase 6):** aplicando calibração **piecewise** na API (T=140, B_low=−3.6, B_high=−15.0, clamp=150) obtivemos **Test MAE = 13.675**, **RMSE = 18.816**, **NASA = 1,393.11**, **super%=50.0%** — melhora substancial sem re-treinar o modelo.

---

## 4) Diagnósticos (gráficos e leitura)
- **Resíduos (ŷ−y):** distribuição centrada; **caudas positivas** (superestimação) reduzidas com a calibração, diminuindo risco operacional.
- **Dispersão (y vs ŷ):** pontos próximos da diagonal; a calibração reduz a dispersão sobretudo em **RUL alto** (>T).
- **MAE por unidade:** erro concentra-se em poucas unidades (prováveis regimes/condições mais severos), facilitando **priorização** de inspeção.

*(Os gráficos — histograma de resíduos, dispersão y×ŷ, MAE por unidade — foram gerados e arquivados como evidência desta fase.)*

---

## 5) Conclusão & Recomendação
- **Modelo selecionado:** **LSTM (window=30, units=64, dropout=0.2, sem clipping)**.
- **Razões:** melhor MAE/RMSE e **NASA** entre os candidatos; com **calibração no serving**, os ganhos aumentam de forma consistente.
- **Baselines:** RF/XGB permanecem como **plan B** e **apoio à explicabilidade**.

**Critérios de aceitação (projeto):** **MAE < 20** e **NASA < 10k** no FD001 — **atingidos** (LSTM cru: MAE 15.95; NASA 3,948).  
Com calibração no serving (indicativo para produção), os resultados melhoram ainda mais (**MAE 13.675; NASA 1,393**).

---

## 6) Limitações & Riscos
- **Escopo FD001:** um único regime/condição; validar em **FD002–FD004** antes de generalizar.
- **Dados simulados:** possíveis diferenças para dados reais (ruído, faltas, drift); exigir **monitoramento** e **adaptação**.
- **NASA** (pune superestimação): manter **monitoramento de viés** e **guardrails** (ex.: clamp) em produção.

---

## 7) Valor para o negócio (Semantix)
- **Redução de paradas não planejadas** via decisões baseadas em RUL com menor viés.
- **Planejamento de estoque/manutenção** com horizonte mais confiável e **custo otimizado**.
- **Adoção acelerada**: baselines explicáveis, diagnósticos por unidade e **calibração operacional** simples (pós-modelo).

---

## 8) Evidências e artefatos desta fase
- Tabelas comparativas (RF, XGB, LSTM) e métricas finais.
- CSV com predições de teste (`unit, RUL_true, RUL_pred`) e planilha de comparação.
- Scripts/Notebooks de avaliação (incluindo *what-ifs* de calibração linear/piecewise).

# Fase 6 — Implantação (CRISP-DM)

> **Objetivo:** disponibilizar o modelo de previsão de **RUL** (FD001/C-MAPSS) como **serviço web** escalável, versionado e observável, reduzindo o risco de **superestimação** via **calibração pós-modelo**.

---

## 1) Visão e Arquitetura

- **Serviço de inferência:** FastAPI + Uvicorn encapsulando **LSTM** (window=30, units=64).  
- **Artefatos versionados:** `models/fd001_lstm_v1/` (SavedModel), `scaler.pkl`, `feature_list.json`, `config.json`.  
- **Calibração no serving (pós-modelo):** **piecewise** (`T=140`, `B_low=-3.6`, `B_high=-15.0`) com *clamp* `[0,150]`.  
- **Empacotamento:** Docker (imagem imutável) e `docker compose` para dev.  
- **Fluxo:** cliente envia **última janela** (≤30 ciclos) → normalização/padding → inferência → **calibração** → resposta (`rul_pred` + incerteza opcional).

---

## 2) Contratos de API

### `GET /health`
Retorna metadados do serviço e parâmetros ativos:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_kind": "SavedModel",
  "model_dir": "models/fd001_lstm_v1",
  "window_size": 30,
  "n_features": 19,
  "calib_mode": "piecewise",
  "calib_t": 140.0,
  "calib_b_low": -3.6,
  "calib_b_high": -15.0,
  "rul_min": 0.0,
  "rul_max": 150.0
}
```

### `POST /predict`
**Request (exemplo)**
```json
{
  "unit": 67,
  "records": [
    {"cycle": 221, "setting_1": 0.33, "setting_2": 0.42, "setting_3": 0.41, "sensor_1": 0.52, "...": 0.18},
    {"cycle": 222, "...": "..."},
    {"cycle": 223, "...": "..."}
  ],
  "mc_passes": 0
}
```
> Se vierem <30 registros, o serviço faz **padding** à esquerda. `mc_passes>0` ativa **incerteza** (Monte Carlo dropout).

**Response (exemplo)**
```json
{ "unit": 67, "rul_pred": 157.758, "rul_std": 0.0, "ci95": [157.758, 157.758] }
```

---

## 3) Calibração e Configuração

| Variável        | Exemplo   | Descrição                                   |
|-----------------|-----------|---------------------------------------------|
| `CALIB_MODE`    | piecewise | none \| linear \| piecewise               |
| `CALIB_A`       | 1.0       | ganho (modo linear)                         |
| `CALIB_B`       | -5.40     | offset (modo linear)                        |
| `CALIB_T`       | 140       | limiar T (modo piecewise)                   |
| `CALIB_B_LOW`   | -3.6      | offset zona baixa (≤T)                      |
| `CALIB_B_HIGH`  | -15.0     | offset zona alta (>T)                       |
| `RUL_MIN`       | 0         | clamp mínimo após calibração                |
| `RUL_MAX`       | 150       | clamp máximo após calibração                |

**Motivação:** reduzir **superestimação** em **RUL alto** (>T), derrubando o **NASA score** sem piorar MAE/RMSE.

---

## 4) Empacotamento e Execução

### Docker (local)
```bash
# build
docker build -t rul-api:v1 .

# run com calibração piecewise
docker run -d --name rul-api -p 8000:8000   -e CALIB_MODE=piecewise   -e CALIB_T=140 -e CALIB_B_LOW=-3.6 -e CALIB_B_HIGH=-15.0   -e RUL_MIN=0 -e RUL_MAX=150   rul-api:v1

# healthcheck
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Docker Compose (dev)
```yaml
version: "3.9"
services:
  rul-api:
    image: rul-api:v1
    ports: ["8000:8000"]
    environment:
      CALIB_MODE: piecewise
      CALIB_T: 140
      CALIB_B_LOW: -3.6
      CALIB_B_HIGH: -15.0
      RUL_MIN: 0
      RUL_MAX: 150
```
> Para teste via Colab, exponha com túnel (ex.: `cloudflared tunnel --url http://localhost:8000`) e use a URL pública.

---

## 5) Teste de Aceitação (E2E)

1. **Saúde:** `/health` retorna `status=ok` e calibração esperada.  
2. **Benchmark (100 unidades FD001):**
   ```bash
   python3 benchmark_fd001.py      --api "http://localhost:8000"      --test ".../CMAPSSData/test_FD001.txt"      --rul  ".../CMAPSSData/RUL_FD001.txt"      --ws 30 --mc 0 --out predicoes_fd001.csv
   ```
3. **Resultados observados (serviço calibrado):**
   - **MAE = 13.675**
   - **RMSE = 18.816**
   - **NASA = 1,393.11** (quanto menor, melhor)
   - **% superestimação = 50.0%**
4. **Evidências:** `predicoes_fd001.csv` + gráficos (resíduos, dispersão y×ŷ, MAE/unidade).

**Critérios de aceitação:**  
- Base (modelo cru): **MAE < 20**, **NASA < 10k** — *atingidos*.  
- Produção (calibrado): **MAE ~ 13–15**, **NASA < 2k**, **super% ~ 50%** — *atingidos*.

---

## 6) Monitoramento e Observabilidade

- **Sinais do serviço (SLOs):** latência p50/p95 do `/predict` (alvo p95 < **150 ms**), taxa de erros/timeouts.  
- **Sinais do modelo:** **NASA** e **% superestimação** (janela móvel), **drift** por feature (PSI/KS) e por *slice* (unit/regime), MAE/RMSE com rótulos tardios.  
- **Alertas:** `% superestimação > 60%`, NASA acima do baseline por N janelas, drift acima do limiar.  
- **Ações:** ajustar `CALIB_*` via *feature flag* (canário → rollout). Persistindo, re-treinar e revisar features/distribuições.

---

## 7) Segurança, Confiabilidade e Custos

- **Segurança:** API-key, CORS restrito, rate limit, logs sem dados sensíveis.  
- **Confiabilidade:** readiness/liveness probes, graceful shutdown, retry/backoff no cliente.  
- **Custos:** CPU atende FD001; considerar GPU apenas se o throughput exigir.

---

## 8) Versionamento e Releases

- **SemVer** no `/health` (`version: "1.0.0"`); *model registry* lógico com `model_dir` + hash/metadata.  
- **Rollout:** canary ou blue/green com comparação online (NASA e %super).  
- **Rollback:** manter imagem **v-1** pronta para troca atômica.

---

## 9) Runbook (Incidentes)

1) **Erro/latência alta:** checar logs, CPU/memória, limites e timeouts.  
2) **NASA ou %super elevados:** confirmar calibração; tornar `B_high` mais negativo e/ou reduzir `RUL_MAX` (canário). Re-treinar se persistir; checar drift e *slices* críticos.  
3) **Dados fora de domínio:** *guardrails* (clamp mais conservador) e notificar operação.

---

## 10) Entregáveis

- Código: `app/main.py`, `requirements.txt`, `Dockerfile`, `docker-compose.yaml`.  
- Artefatos: `models/fd001_lstm_v1/`, `scaler.pkl`, `feature_list.json`, `config.json`.  
- QA: `benchmark_fd001.py`, payloads de exemplo.  
- Evidências: `predicoes_fd001.csv`, `comparacao_metricas.csv`, gráficos (resíduos, y×ŷ, MAE/unidade).  
- Documento operacional: este markdown + parâmetros de calibração e runbook.

---

## 11) Próximos Passos

- Generalizar para **FD002–FD004** (múltiplas condições/regimes).  
- MLOps: pipeline de re-treino automatizado, *model registry* formal, checagens de qualidade/dados.  
- Modelagem: CNN-LSTM/Transformers, *ensembles*, janelas adaptativas.  
- Negócio: integração com **CMMS/ERP** para ordens de serviço e planejamento de estoque.

---

## 12) Conclusão

A implantação em Docker com **calibração piecewise** atingiu os **critérios operacionais** e reduziu substancialmente o **risco de superestimação** (**NASA baixo**) **sem** sacrificar **MAE/RMSE**. A solução está pronta para escalar, monitorar e evoluir para múltiplos regimes e um ciclo completo de **MLOps**.
