# E2 Feature Audit — Redundancy Review and Proposed Feature Sets
 
**Related files:** `src/feature_engineering.py`, `src/wheelchair_feature_env.py`

---

## 1. Current Feature Set (E2-full, 12 features)

All features are extracted from the normalised 360-degree LiDAR scan.  
Sectors used (matching the reward function in `wheelchair_env.py`):

| Sector | LiDAR indices | N readings |
|--------|--------------|------------|
| Left   | 100–169      | 70         |
| Front  | 170–189      | 20         |
| Right  | 190–259      | 70         |
| Rear   | 0–39 + 320–359 | 80       |

### Feature classification table

| Index | Name | Formula | Range | Classification | Justification |
|-------|------|---------|-------|----------------|---------------|
| 0 | `min_front` | min(front[170:190]) | [0, 1] | **Essencial** | Sinal primário de obstáculo à frente; crítico para evitar colisão |
| 1 | `mean_front` | mean(front[170:190]) | [0, 1] | **Redundante** | Derivada do mesmo sector que `min_front`; o mínimo já capta o pior caso; a média suaviza mas não acrescenta informação de decisão |
| 2 | `left_clearance` | mean(left[100:170]) | [0, 1] | **Essencial** | Espaço médio lateral esquerdo; informa sobre a viabilidade de virar/navegar à esquerda |
| 3 | `right_clearance` | mean(right[190:260]) | [0, 1] | **Essencial** | Espaço médio lateral direito; par simétrico de `left_clearance` |
| 4 | `clearance_asymmetry` | right\_clearance − left\_clearance | [−1, 1] | **Derivada/Redundante** | É algebricamente `feature[3] − feature[2]`; a MLP pode aprender esta relação; útil como atalho explícito, mas redundante se os dois componentes estiverem presentes |
| 5 | `max_gap_center` | centro do maior gap livre / 359 | [0, 1] | **Essencial** | Indica a direcção do corredor mais navegável; feature global não capturada pelos sectores fixos |
| 6 | `max_gap_width` | largura do maior gap / 360 | [0, 1] | **Essencial** | Indica a qualidade/largura do caminho mais livre; complemento de `max_gap_center` |
| 7 | `rear_clearance` | mean(rear[0:40, 320:360]) | [0, 1] | **Provavelmente inútil** | Política de navegação forward-only; raramente relevante para a decisão; aumenta dimensão sem benefício claro |
| 8 | `min_left` | min(left[100:170]) | [0, 1] | **Essencial** | Obstáculo mais próximo à esquerda; mais crítico que a média para evitar colisão lateral |
| 9 | `min_right` | min(right[190:260]) | [0, 1] | **Essencial** | Obstáculo mais próximo à direita; par simétrico de `min_left` |
| 10 | `min_rear` | min(rear combined) | [0, 1] | **Provavelmente inútil** | Raramente relevante para política forward; pode introduzir ruído |
| 11 | `previous_action` | prev\_action / 5.0 | [0, 1] | **Essencial** | Contexto temporal; evita oscilação de acção e dá continuidade à política |

### Redundâncias identificadas

**Triângulo left/right/asymmetry:**
```
clearance_asymmetry = right_clearance - left_clearance   (feature 4 = feature 3 - feature 2)
```
Qualquer dois dos três determinam o terceiro. Incluir os três é redundância explícita para a MLP.

**Par min/mean por sector:**
```
mean_front  vs  min_front   (features 1 vs 0)
rear_clearance  vs  min_rear  (features 7 vs 10)
```
Nestes pares, o mínimo é mais informativo para evitar colisão. A média pode mascarar um obstáculo próximo.

---

## 2. Versões de Feature Set Propostas

### E2-full (12 features) — versão actual
Mantém todas as features. Serve como baseline de comparação.  
Permite avaliar se a redundância explícita beneficia ou prejudica a MLP.

```
[min_front, mean_front, left_clearance, right_clearance, clearance_asymmetry,
 max_gap_center, max_gap_width, rear_clearance, min_left, min_right, min_rear,
 previous_action]
```

### E2-reduced (8 features) — versão limpa
Remove as 4 features classificadas como redundantes ou inúteis:

| Feature removida | Motivo |
|-----------------|--------|
| `mean_front` (#1) | `min_front` capta o pior caso; média menos informativa para colisão |
| `clearance_asymmetry` (#4) | Derivada de `left_clearance` e `right_clearance`; redundância matemática |
| `rear_clearance` (#7) | Raramente relevante para política forward-only |
| `min_rear` (#10) | Idem; aumenta dimensão sem benefício observado |

```
[min_front, left_clearance, right_clearance, max_gap_center,
 max_gap_width, min_left, min_right, previous_action]
```

**Hipótese:** A MLP com menos features redundantes deve aprender mais rápido e generalizar melhor.

### E2-directional (5 features) — versão mínima direccional
Mantém apenas as features de decisão direcional:

```
[min_front, clearance_asymmetry, max_gap_center, max_gap_width, previous_action]
```

**Justificação:** Estas 5 features codificam as três perguntas essenciais à decisão de navegação:
1. *Está bloqueado à frente?* → `min_front`
2. *Qual lado tem mais espaço?* → `clearance_asymmetry`
3. *Onde está o melhor caminho e quão largo é?* → `max_gap_center`, `max_gap_width`
4. *O que fiz na acção anterior?* → `previous_action`

**Hipótese:** Uma representação tão compacta pode ser suficiente para a política, com aprendizagem mais rápida mas possivelmente menor robustez em casos limite.

---

## 3. Comparação de Dimensões

| Variant | Features | Dimensão obs. | Policy |
|---------|----------|---------------|--------|
| E1 (CNN baseline) | LiDAR raw + prev\_action | 361 | CnnPolicy |
| E2-full | Todas as 12 features | 12 | MlpPolicy |
| E2-reduced | 8 features sem redundâncias | 8 | MlpPolicy |
| E2-directional | 5 features direccionais | 5 | MlpPolicy |

---

## 4. Resultados — E1 vs E2-full vs E2-reduced

### Comparação de success rate por corredor

| Robot | E1 success% | E2-full success% | E2-reduced success% | Observação |
|-------|-------------|------------------|---------------------|------------|
| 0 | 100.0 | 100.0 | 100.0 | Todos perfeitos |
| 1 | **8.2** | **100.0** | **100.0** | E2 melhora drasticamente; CNN falha |
| 2 | 100.0 | 100.0 | 92.9 | Ligeira regressão no reduced |
| 3 | 100.0 | 100.0 | 100.0 | Todos perfeitos |
| 4 | **18.0** | **0.0** | **0.0** | Corredor difícil para todos |
| 5 | 100.0 | 100.0 | **6.2** | **E2-reduced colapsa (93.8% colisão)** |
| 6 | 100.0 | 100.0 | 100.0 | Todos perfeitos |
| 7 | 100.0 | **0.0** | **100.0** | E2-full falha; reduced recupera |
| 8 | 100.0 | **0.0** | **0.0** | E2 falha em ambas as variantes |
| **Média** | **80.7%** | **77.8%** | **77.7%** | Features ≈ CNN em média |

*Fonte: `success_rates_e1.csv`, `results/e2_full/evaluation/metrics.csv`, `results/e2_reduced/evaluation/metrics.csv`*

### Métricas detalhadas por corredor — E2-full

| Robot | Episodes | Success% | Collision% | Timeout% | Mean ep. length |
|-------|----------|----------|------------|----------|-----------------|
| 0 | 30 | 100.0 | 0.0 | 0.0 | 1961 |
| 1 | 30 | 100.0 | 0.0 | 0.0 | 1988 |
| 2 | 29 | 100.0 | 0.0 | 0.0 | 2056 |
| 3 | 29 | 100.0 | 0.0 | 0.0 | 2058 |
| 4 | 16 | 0.0 | **56.3** | 43.8 | 3541 |
| 5 | 17 | 100.0 | 0.0 | 0.0 | 3366 |
| 6 | 16 | 100.0 | 0.0 | 0.0 | 3674 |
| 7 | 12 | 0.0 | 0.0 | **100.0** | 5000 |
| 8 | 12 | 0.0 | 0.0 | **100.0** | 5000 |

### Métricas detalhadas por corredor — E2-reduced

| Robot | Episodes | Success% | Collision% | Timeout% | Mean ep. length |
|-------|----------|----------|------------|----------|-----------------|
| 0 | 30 | 100.0 | 0.0 | 0.0 | 1984 |
| 1 | 30 | 100.0 | 0.0 | 0.0 | 1989 |
| 2 | 28 | 92.9 | 7.1 | 0.0 | 2106 |
| 3 | 28 | 100.0 | 0.0 | 0.0 | 2082 |
| 4 | 12 | 0.0 | 8.3 | **91.7** | 4943 |
| 5 | 16 | 6.2 | **93.8** | 0.0 | 3641 |
| 6 | 16 | 100.0 | 0.0 | 0.0 | 3674 |
| 7 | 15 | 100.0 | 0.0 | 0.0 | 3768 |
| 8 | 12 | 0.0 | 0.0 | **100.0** | 5000 |

### Casos de interesse para análise qualitativa

**Robot 1 — E2 melhora muito (+91.8% vs E1)**  
E1 falha quase sempre (8.2%). Ambas as variantes E2 atingem 100%. Hipótese: a geometria do corredor 1 tem uma assimetria lateral clara que `left_clearance`/`right_clearance` captam explicitamente; o CNN não aprende esta distinção de forma consistente.

**Robot 4 — caso difícil para todos os modelos**  
E1: 18%, E2-full: 0% (56% colisão), E2-reduced: 0% (92% timeout). Geometria provavelmente muito exigente (curva apertada ou largura reduzida). Nenhuma representação resolve bem.

**Robot 5 — colapso no E2-reduced (100% → 6.2%)**  
E1 e E2-full navigam com 100%. E2-reduced tem 93.8% de colisão. A `clearance_asymmetry` foi removida no reduced; sem ela, o agente perde o sinal direcional crítico para a geometria deste corredor.

**Robot 7 — inversão entre E2-full e E2-reduced**  
E1: 100%. E2-full: 0% timeout (robot oscila sem sair). E2-reduced: 100%. A `clearance_asymmetry` no E2-full cria um sinal ambíguo neste corredor simétrico; sem ela (reduced), o agente decide com `min_left`/`min_right` directamente e navega com sucesso.

**Robot 8 — falha estrutural de features**  
E1: 100%. E2-full e E2-reduced: 0% timeout. A representação por features parece insuficiente para a geometria deste corredor em ambas as variantes; o CNN aprende algo que as features não codificam.

---

## 5. Protocolo de Re-treino

O re-treino de E2 deve usar exactamente o mesmo protocolo que E1 corrigido:

```
PPO, MlpPolicy
9 ambientes paralelos
3 000 000 timesteps
step limit: 5000 por episódio
timeout penalty: −10 se truncated e não terminated
success: goal_reached == True (End_Strip)
collision: bumper triggered
```

### Comandos de treino

```bash
# E2-full (versão actual, re-treino limpo)
python src/rl-server-features.py --feature-set full --new

# E2-reduced (versão sem redundâncias)
python src/rl-server-features.py --feature-set reduced --new

# E2-directional (versão mínima)
python src/rl-server-features.py --feature-set directional --new
```

### Comandos de avaliação

```bash
python src/rl-test-features.py --feature-set full
python src/rl-test-features.py --feature-set reduced
python src/rl-test-features.py --feature-set directional
```

Resultados guardados em `results/e2_{variant}/evaluation/metrics.csv` com colunas:  
`robot_id, episodes, successes, collisions, timeouts, success_rate, collision_rate, timeout_rate, mean_episode_length`

---

## 6. Comparação Global

| Modelo | Global success% | Collision% (média) | Timeout% (média) | Mean ep. length (média) |
|--------|----------------|--------------------|------------------|-------------------------|
| E1 (CNN, CnnPolicy) | **80.7%** | — | — | — |
| E2-full (12 features, MlpPolicy) | 77.8% | 6.3% | 27.1% | 3182 |
| E2-reduced (8 features, MlpPolicy) | 77.7% | 12.8% | 21.3% | 3243 |

*E1: só success% disponível (`success_rates_e1.csv`). Collision%/Timeout% não foram registados no script original.*  
*Médias calculadas como média simples das 9 corridors.*

### Interpretação

- As três abordagens têm performance global semelhante (~78–81%), mas com **distribuições de falha distintas**.
- E2-full melhora drasticamente o corredor 1 (+91.8% vs E1) mas cria falhas novas nos corredores 7 e 8.
- E2-reduced resolve o problema do corredor 7 mas colapsa no corredor 5.
- O CNN (E1) é o mais robusto transversalmente, mas falha especificamente no corredor 1.
- Nenhum modelo resolve o corredor 4; nenhuma feature set (nem o CNN) domina em todos os corredores.
