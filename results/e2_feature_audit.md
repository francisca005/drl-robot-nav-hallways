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

## 4. Resultados Finais — E1 vs E2 (todas as variantes)

> Resultados obtidos após correcção do bug `min_distance_from_start_for_goal` (2.0 → 0.5)  
> e re-treino completo das 3 variantes E2 com 3 000 000 timesteps cada.

### Comparação de success rate por corredor

| Robot | E1 (CNN) | E2-full (12f) | E2-reduced (8f) | E2-directional (5f) | Observação |
|-------|----------|---------------|-----------------|---------------------|------------|
| 0 | 100.0 | 100.0 | 100.0 | 100.0 | Todos perfeitos |
| 1 | **8.2** | **100.0** | **100.0** | **100.0** | E2 resolve; CNN falha |
| 2 | 100.0 | 100.0 | 100.0 | 100.0 | Todos perfeitos |
| 3 | 100.0 | 100.0 | 100.0 | 100.0 | Todos perfeitos |
| 4 | 18.0 | **0.0** | **0.0** | **0.0** | Corredor difícil — todos falham |
| 5 | 100.0 | 100.0 | **0.0** | 100.0 | **E2-reduced colapsa (100% colisão)** |
| 6 | 100.0 | 100.0 | 100.0 | **0.0** | **E2-directional falha (100% timeout)** |
| 7 | 100.0 | 100.0 | **60.0** | 100.0 | **E2-reduced instável (40% colisão)** |
| 8 | 100.0 | 100.0 | 100.0 | 100.0 | Todos resolvem após correcção do bug |
| **Média** | **80.7%** | **88.9%** | **73.3%** | **77.8%** | E2-full é a melhor variante |

### Convergência durante o treino

| Variante | 1ª vez 100% | Success rate final | Tempo treino |
|----------|-------------|-------------------|--------------|
| E2-full (12f) | ~2.1M steps | **100%** | ~3.2h |
| E2-reduced (8f) | nunca | 88% | ~3.1h |
| E2-directional (5f) | ~2.1M steps | **100%** | ~3.1h |

### Métricas detalhadas — E2-full (12f)

| Robot | Episodes | Success% | Collision% | Timeout% | Mean ep. length |
|-------|----------|----------|------------|----------|-----------------|
| 0 | 30 | 100.0 | 0.0 | 0.0 | 1964 |
| 1 | 29 | 100.0 | 0.0 | 0.0 | 2020 |
| 2 | 29 | 100.0 | 0.0 | 0.0 | 2015 |
| 3 | 29 | 100.0 | 0.0 | 0.0 | 2018 |
| 4 | 13 | 0.0 | **84.6** | 15.4 | 4523 |
| 5 | 17 | 100.0 | 0.0 | 0.0 | 3336 |
| 6 | 16 | 100.0 | 0.0 | 0.0 | 3674 |
| 7 | 16 | 100.0 | 0.0 | 0.0 | 3609 |
| 8 | 17 | 100.0 | 0.0 | 0.0 | 3419 |

### Métricas detalhadas — E2-reduced (8f)

| Robot | Episodes | Success% | Collision% | Timeout% | Mean ep. length |
|-------|----------|----------|------------|----------|-----------------|
| 0 | 29 | 100.0 | 0.0 | 0.0 | 2037 |
| 1 | 29 | 100.0 | 0.0 | 0.0 | 2004 |
| 2 | 26 | 100.0 | 0.0 | 0.0 | 2240 |
| 3 | 29 | 100.0 | 0.0 | 0.0 | 2050 |
| 4 | 43 | 0.0 | **100.0** | 0.0 | 1368 |
| 5 | 16 | 0.0 | **100.0** | 0.0 | 3589 |
| 6 | 16 | 100.0 | 0.0 | 0.0 | 3691 |
| 7 | 15 | 60.0 | **40.0** | 0.0 | 3852 |
| 8 | 17 | 100.0 | 0.0 | 0.0 | 3374 |

### Métricas detalhadas — E2-directional (5f)

| Robot | Episodes | Success% | Collision% | Timeout% | Mean ep. length |
|-------|----------|----------|------------|----------|-----------------|
| 0 | 30 | 100.0 | 0.0 | 0.0 | 1967 |
| 1 | 29 | 100.0 | 0.0 | 0.0 | 2023 |
| 2 | 29 | 100.0 | 0.0 | 0.0 | 2019 |
| 3 | 29 | 100.0 | 0.0 | 0.0 | 2016 |
| 4 | 28 | 0.0 | **85.7** | 14.3 | 2101 |
| 5 | 18 | 100.0 | 0.0 | 0.0 | 3312 |
| 6 | 12 | 0.0 | 0.0 | **100.0** | 5000 |
| 7 | 16 | 100.0 | 0.0 | 0.0 | 3583 |
| 8 | 17 | 100.0 | 0.0 | 0.0 | 3407 |

### Casos de interesse para análise qualitativa

**Robot 1 — E2 resolve onde E1 falha (+91.8%)**  
E1: 8.2%. Todas as variantes E2 atingem 100%. A geometria do corredor 1 tem uma assimetria lateral que `left_clearance`/`right_clearance` captam explicitamente; a CNN não aprende esta distinção de forma consistente.

**Robot 4 — corredor difícil para todos os modelos**  
E1: 18% (único com algum sucesso). E2-full: 0% (84.6% colisão). E2-reduced: 0% (100% colisão). E2-directional: 0% (85.7% colisão). Geometria muito exigente; nenhuma representação resolve.

**Robot 5 — colapso exclusivo no E2-reduced**  
E1, E2-full e E2-directional: 100%. E2-reduced: 0% (100% colisão). A remoção de `clearance_asymmetry` elimina o sinal direcional crítico para este corredor. O E2-directional inclui `clearance_asymmetry` e navega sem problemas — confirma que esta feature é essencial aqui.

**Robot 6 — falha exclusiva no E2-directional**  
E1, E2-full e E2-reduced: 100%. E2-directional: 0% (100% timeout). O directional não tem `min_left`/`min_right` separados — perde informação de proximidade lateral que é necessária neste corredor.

**Robot 7 — E2-reduced instável**  
E1, E2-full e E2-directional: 100%. E2-reduced: 60% (40% colisão). Sem `clearance_asymmetry`, o agente falha intermitentemente neste corredor.

**Robot 8 — resolvido após correcção do bug**  
Resultados anteriores (0% timeout em E2-full e E2-reduced) deviam-se a um bug na condição de sucesso (`min_distance_from_start_for_goal = 2.0`): a faixa verde fica geometricamente perto da posição inicial, tornando impossível satisfazer a condição. Após correcção para 0.5, todas as variantes atingem 100%.

---

## 5. Protocolo de Re-treino (concluído)

Re-treino efectuado com o código corrigido (`min_distance_from_start_for_goal = 0.5`):

```
PPO, MlpPolicy
9 ambientes paralelos
3 000 000 timesteps
step limit: 5000 por episódio
timeout penalty: −10 se truncated e não terminated
success: goal_reached == True (End_Strip) AND steps ≥ 20 AND dist_from_start ≥ 0.5
collision: bumper triggered
```

```bash
python src/rl-server-features.py --feature-set full --new
python src/rl-server-features.py --feature-set reduced --new
python src/rl-server-features.py --feature-set directional --new

python src/rl-test-features.py --feature-set full
python src/rl-test-features.py --feature-set reduced
python src/rl-test-features.py --feature-set directional
```

---

## 6. Comparação Final

| Modelo | Mean success% | Mean collision% | Mean timeout% | Mean ep. length |
|--------|--------------|-----------------|---------------|-----------------|
| E1 (CNN) | 80.7% | — | — | — |
| E2-full (12f) | **88.9%** | 9.4% | 1.7% | 3064 |
| E2-reduced (8f) | 73.3% | 26.7% | 0.0% | 2912 |
| E2-directional (5f) | 77.8% | 9.5% | 12.7% | 2825 |

**Conclusão:** O E2-full (12 features) é a melhor variante, superando o E1 (CNN) em 8.2 p.p. de média global. A hipótese de que remover redundâncias melhoraria o desempenho não se confirmou — o E2-reduced é a pior variante. O E2-directional com apenas 5 features é surpreendentemente competitivo mas falha num corredor específico (robot 6) por falta de informação lateral detalhada.
