# 🎓 Exemplos de Uso - Dashboard Interativo

## Exemplo 1: Comparação de Diagnósticos entre Gêneros

### Objetivo
Verificar se a distribuição de diagnósticos varia significativamente entre homens e mulheres.

### Passos
1. Abra o dashboard e vá para **Visão Geral**
2. Na seção "🎯 Filtros de Estratificação":
   - Deixe "Faixa Etária" como "✨ Todos"
   - Selecione "👨 Masculino" no filtro de gênero
3. Observe o gráfico "Distribuição de Diagnósticos"
4. Mude o filtro para "👩 Feminino"
5. Compare os dois resultados

### Resultado Esperado
```
Masculino:
- Dengue: 45%
- Gripe: 30%
- Chikungunya: 25%

Feminino:
- Dengue: 50%
- Gripe: 25%
- Chikungunya: 25%
```

### Insight
Se há diferenças, isso pode indicar que o gênero é um fator importante na
suscetibilidade a determinadas doenças climáticas.

---

## Exemplo 2: Análise por Faixa Etária

### Objetivo
Determinar qual faixa etária é mais afetada por doenças relacionadas ao clima.

### Passos
1. Em "Visão Geral", acesse os filtros
2. Mantenha "👤 Gênero" como "✨ Todos"
3. Altere "🎂 Faixa Etária" para cada opção:
   - Primeiro: "👶 Crianças (0-12)"
   - Depois: "👨 Adultos (18-59)"
   - Por fim: "👴 Idosos (60+)"
4. Observe como a distribuição muda

### Análise de Dados
```
Métrica               | Crianças | Adultos | Idosos
Total de Casos        | 800      | 2500    | 1200
Taxa de Dengue        | 35%      | 50%     | 55%
Taxa de Gripe         | 55%      | 30%     | 40%
Caso Mais Grave       | Gripe    | Dengue  | Dengue
```

### Interpretação
- Crianças: Mais suscetíveis à gripe
- Adultos: Maior incidência de dengue
- Idosos: Pior prognóstico com dengue

---

## Exemplo 3: Estratificação por Gênero na EDA

### Objetivo
Investigar se sintomas respiratórios variam entre gêneros com mudanças climáticas.

### Passos
1. Vá para **Análise Exploratória**
2. Na seção "🌤️ Explorador Climático Interativo":
   - Deixe filtros climáticos como "Todos"
   - Selecione "👤 Gênero" = "👨 Masculino"
3. Observe o gráfico "Regressão: Vento vs Sintomas Respiratórios"
4. Mude "Gênero" para "👩 Feminino"
5. Compare os coeficientes de correlação

### Resultado Esperado
```
Masculino:
- Correlação: 0.45 (positiva moderada)
- Interpretação: Aumenta vento → Aumenta sintomas

Feminino:
- Correlação: 0.62 (positiva forte)
- Interpretação: Aumenta vento → Aumenta mais sintomas
```

### Conclusão
Mulheres podem ser mais sensíveis a aumentos de velocidade do vento em
relação a sintomas respiratórios.

---

## Exemplo 4: Combinação de Filtros

### Objetivo
Focar em um subgrupo específico: "Mulheres Adultas".

### Passos
1. Em "Visão Geral", selecione:
   - "👤 Gênero" = "👩 Feminino"
   - "🎂 Faixa Etária" = "👨 Adultos (18-59)"
2. Todos os gráficos agora mostram apenas dados desta população
3. Compare com:
   - "👨 Masculino" + "👨 Adultos (18-59)"
4. Identifique diferenças específicas

### Análise Comparativa
```
                        | Fem. Adultas | Masc. Adultos
Casos Totais            | 1200         | 1300
Dengue (%)              | 52%          | 48%
Gripe (%)               | 20%          | 35%
Chikungunya (%)         | 28%          | 17%

Temperatura Média       | 26°C         | 26°C
Umidade Média           | 0.68         | 0.68
Vento Médio             | 8 km/h       | 8 km/h
```

### Observação
Com mesmas condições climáticas, mulheres adultas têm maior incidência
de Chikungunya, enquanto homens adultos têm mais gripe.

---

## Exemplo 5: Análise de Matriz de Correlação

### Objetivo
Entender como variáveis se relacionam dentro de cada gênero.

### Passos
1. Em "Análise Exploratória", vá até "Matriz de Correlação (Top Features)"
2. Selecione "👤 Gênero" = "👨 Masculino"
3. Observe quais variáveis têm forte correlação
4. Mude para "👩 Feminino" e compare

### Exemplo de Interpretação
```
Masculino:
- Temperatura ↔ Dengue: 0.68 (forte positiva)
- Umidade ↔ Gripe: 0.45 (moderada positiva)
- Idade ↔ Severity: 0.52 (moderada positiva)

Feminino:
- Temperatura ↔ Dengue: 0.72 (forte positiva)
- Umidade ↔ Gripe: 0.38 (fraca positiva)
- Idade ↔ Severity: 0.48 (moderada positiva)
```

### Conclusão
Mulheres mostram relação mais forte entre temperatura e dengue,
sugerindo maior sensibilidade ao fator térmico.

---

## Exemplo 6: Exploração de Perfis Climáticos

### Objetivo
Identificar qual perfil climático afeta mais cada grupo demográfico.

### Passos
1. Em "Análise Exploratória", filtros climáticos:
   - "🌡️ Temperatura" = "🔥 Alto (>25°C)"
   - "💧 Umidade" = "💦 Alta (>0.7)"
   - "💨 Vento" = "🌪️ Alto (>15 km/h)"
   - "👤 Gênero" = "👨 Masculino"
2. Observe incidência neste perfil
3. Repita com outro gênero
4. Compare resultados

### Cenário
```
PERFIL: Quente, Úmido, Ventoso + Masculino
- Total de Casos: 150
- Dengue: 70 (47%)
- Gripe: 45 (30%)
- Chikungunya: 35 (23%)

PERFIL: Quente, Úmido, Ventoso + Feminino
- Total de Casos: 140
- Dengue: 60 (43%)
- Gripe: 32 (23%)
- Chikungunya: 48 (34%)
```

### Insight
Em condições de calor, umidade e vento altos, mulheres têm maior
incidência de Chikungunya, enquanto homens têm mais dengue.

---

## Exemplo 7: Análise Temporal por Gênero

### Objetivo
Examinar como a distribuição de diagnósticos varia em diferentes condições.

### Passos
1. Vá para "Análise Exploratória"
2. Use controles climáticos para simular "estações":
   
   **Estação Quente:**
   - Temperatura: "Alto"
   - Umidade: "Alta"
   - Resultado: Quantos casos?
   
   **Estação Fria:**
   - Temperatura: "Baixo"
   - Umidade: "Baixa"
   - Resultado: Quantos casos?

3. Faça para cada gênero

### Comparação
```
                | Quente/Úmido | Frio/Seco
Masculino Cases | 450          | 280
Feminino Cases  | 420          | 310
Razão M/F       | 1.07         | 0.90
```

---

## Checklist de Exploração

Quando usar o dashboard, explore:

- [ ] Diferenças de diagnósticos por gênero
- [ ] Padrões etários de incidência
- [ ] Correlações por grupo demográfico
- [ ] Impacto de temperatura isoladamente
- [ ] Impacto de umidade isoladamente
- [ ] Efeito combinado de fatores climáticos
- [ ] Sintomas mais frequentes por subgrupo
- [ ] Variações de severidade
- [ ] Padrões sazonais simulados
- [ ] Outliers ou casos incomuns

---

## 💡 Dicas Práticas

### Dica 1: Screenshot para Comparação
- Tome screenshot com um filtro
- Mude os filtros
- Abra ambos lado a lado
- Identifique diferenças visuais

### Dica 2: Anotações
- Anote números-chave (percentuais, médias)
- Compare em diferentes filtros
- Procure por padrões consistentes

### Dica 3: Hipóteses
- Forme hipóteses antes de filtrar
- Teste com dados
- Confirme ou refute

### Dica 4: Visualização
- Foque em uma coisa por vez
- Use filtros um a um
- Depois combine filtros
- Observe efeitos emergentes

---

## 📊 Métricas Importantes

Ao explorar, procure por:

| Métrica | O que indica |
|---------|-------------|
| Mudança em % casos | Diferentes suscetibilidades |
| Correlação forte | Relação causal provável |
| Outliers | Casos especiais/exceções |
| Padrão consistente | Tendência confiável |
| Variabilidade alta | Heterogeneidade no grupo |

---

**Último Conselho**: Combine análises quantitativas (números) com análises
visuais (gráficos) para melhor compreensão dos dados!
