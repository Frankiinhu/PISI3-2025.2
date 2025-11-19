"""
Funções auxiliares para visualizações SHAP
Suporta: feature importance, barras multiclasse, beeswarm plots e force plots
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_shap_feature_importance_bar(
    shap_values,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    top_n: int = 15,
    title: str = "Feature Importance (SHAP Values)"
) -> go.Figure:
    """
    Cria gráfico de barras de feature importance baseado em SHAP values.
    
    Args:
        shap_values: SHAP values (lista para multiclasse ou array)
        feature_names: Lista com nomes das features
        class_names: Nomes das classes (opcional)
        top_n: Número de top features para mostrar
        title: Título do gráfico
        
    Returns:
        Plotly Figure
    """
    try:
        # Calcular importância média absoluta
        if isinstance(shap_values, list):
            # Multiclasse: média entre classes
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            if len(shap_values.shape) == 3:
                # (samples, features, classes)
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                # (samples, features)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Ordenar e pegar top N
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = mean_abs_shap[top_indices]
        
        # Criar gráfico
        fig = go.Figure(go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importances,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Importância<br>|SHAP|")
            ),
            text=[f'{val:.4f}' for val in top_importances],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importância: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='|SHAP Value| Médio',
            yaxis_title='Features',
            height=max(400, top_n * 30),
            template='plotly_dark',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar feature importance bar: {e}")
        return go.Figure()


def create_shap_multiclass_bar(
    shap_values: List[np.ndarray],
    feature_names: List[str],
    class_names: List[str],
    top_n: int = 10,
    title: str = "SHAP Feature Importance por Classe"
) -> go.Figure:
    """
    Cria gráfico de barras agrupadas mostrando importância SHAP por classe.
    
    Args:
        shap_values: Lista de SHAP values (um array por classe)
        feature_names: Lista com nomes das features
        class_names: Lista com nomes das classes
        top_n: Número de features para mostrar
        title: Título do gráfico
        
    Returns:
        Plotly Figure
    """
    try:
        if not isinstance(shap_values, list):
            logger.warning("shap_values deve ser uma lista para multiclass plot")
            return go.Figure()
        
        n_classes = len(shap_values)
        
        # Calcular importância por classe
        importances_by_class = []
        for class_idx in range(n_classes):
            class_importance = np.abs(shap_values[class_idx]).mean(axis=0)
            importances_by_class.append(class_importance)
        
        # Encontrar top features pela importância média geral
        mean_importance = np.mean(importances_by_class, axis=0)
        top_indices = np.argsort(mean_importance)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        
        # Criar figura
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for class_idx in range(n_classes):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f'Classe {class_idx}'
            class_values = [importances_by_class[class_idx][i] for i in top_indices]
            
            fig.add_trace(go.Bar(
                name=class_name,
                x=top_features,
                y=class_values,
                marker_color=colors[class_idx % len(colors)],
                hovertemplate=f'<b>{class_name}</b><br>%{{x}}<br>Importância: %{{y:.4f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Features',
            yaxis_title='|SHAP Value| Médio',
            barmode='group',
            height=600,
            template='plotly_dark',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            xaxis={'tickangle': -45}
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar multiclass bar plot: {e}")
        return go.Figure()


def create_shap_beeswarm(
    shap_values,
    shap_data: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
    title: str = "SHAP Beeswarm Plot"
) -> go.Figure:
    """
    Cria beeswarm plot mostrando distribuição dos SHAP values.
    Cada ponto é uma amostra, cor indica valor da feature.
    
    Args:
        shap_values: SHAP values
        shap_data: Dados originais (para colorir pontos)
        feature_names: Lista com nomes das features
        top_n: Número de features para mostrar
        title: Título do gráfico
        
    Returns:
        Plotly Figure
    """
    try:
        # Calcular importância para ordenar features
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            if len(shap_values.shape) == 3:
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Top features
        top_indices = np.argsort(mean_abs_shap)[-top_n:]
        
        fig = go.Figure()
        
        for idx, feat_idx in enumerate(top_indices):
            feat_name = feature_names[feat_idx]
            
            # Extrair SHAP values para essa feature
            if isinstance(shap_values, list):
                feat_shap = np.mean([sv[:, feat_idx] for sv in shap_values], axis=0)
            else:
                if len(shap_values.shape) == 3:
                    feat_shap = shap_values[:, feat_idx, :].mean(axis=1)
                else:
                    feat_shap = shap_values[:, feat_idx]
            
            # Valores da feature (para cor)
            if len(shap_data.shape) > 1:
                feat_values = shap_data[:, feat_idx]
            else:
                feat_values = np.ones(len(feat_shap)) * 0.5
            
            # Adicionar jitter vertical
            y_positions = np.full(len(feat_shap), idx)
            jitter = np.random.normal(0, 0.15, len(feat_shap))
            y_jittered = y_positions + jitter
            
            fig.add_trace(go.Scatter(
                x=feat_shap,
                y=y_jittered,
                mode='markers',
                marker=dict(
                    size=5,
                    color=feat_values,
                    colorscale='RdBu_r',
                    opacity=0.6,
                    line=dict(width=0.3, color='white'),
                    showscale=(idx == 0),
                    colorbar=dict(
                        title='Valor<br>Feature',
                        x=1.02,
                        len=0.7
                    ) if idx == 0 else None
                ),
                name=feat_name,
                showlegend=False,
                hovertemplate=f'<b>{feat_name}</b><br>SHAP: %{{x:.3f}}<br>Valor: %{{marker.color:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='SHAP Value (impacto na predição)',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(top_indices))),
                ticktext=[feature_names[i] for i in top_indices],
                title=''
            ),
            height=max(500, top_n * 35),
            template='plotly_dark',
            hovermode='closest'
        )
        
        # Adicionar linha vertical em x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar beeswarm plot: {e}")
        return go.Figure()


def create_shap_force_plot(
    shap_values,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: List[str],
    predicted_class: str,
    sample_idx: int = 0,
    class_idx: Optional[int] = None,
    max_display: int = 10
) -> go.Figure:
    """
    Cria force plot (waterfall) para explicação local de uma predição individual.
    
    Args:
        shap_values: SHAP values (lista ou array)
        base_value: Valor base (expectativa do modelo)
        feature_values: Valores das features para a amostra
        feature_names: Lista com nomes das features
        predicted_class: Classe predita
        sample_idx: Índice da amostra
        class_idx: Índice da classe (para multiclasse)
        max_display: Número máximo de features para mostrar
        
    Returns:
        Plotly Figure (waterfall plot)
    """
    try:
        # Selecionar SHAP values da amostra e classe específica
        if isinstance(shap_values, list):
            if class_idx is None:
                class_idx = 0
            shap_vals = shap_values[class_idx][sample_idx]
        elif len(shap_values.shape) == 3:
            if class_idx is None:
                class_idx = 0
            shap_vals = shap_values[sample_idx, :, class_idx]
        else:
            shap_vals = shap_values[sample_idx]
        
        # Ordenar por magnitude absoluta
        abs_importance = np.abs(shap_vals)
        top_indices = np.argsort(abs_importance)[-max_display:][::-1]
        
        # Pegar valores relevantes
        top_shap = shap_vals[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        top_values = feature_values[top_indices]
        
        # Criar labels
        labels = ['Valor Base'] + [f"{feat}<br>= {val:.2f}" for feat, val in zip(top_features, top_values)] + ['Predição']
        
        # Calcular valores acumulados
        cumulative = [base_value]
        for val in top_shap:
            cumulative.append(cumulative[-1] + val)
        
        # Valores para waterfall
        values = [base_value] + list(top_shap) + [cumulative[-1]]
        measures = ['absolute'] + ['relative'] * len(top_shap) + ['total']
        
        # Cores (vermelho negativo, verde positivo)
        colors = ['lightgray']
        for val in top_shap:
            colors.append('#EF553B' if val < 0 else '#00CC96')
        colors.append('#636EFA')
        
        # Criar waterfall
        fig = go.Figure(go.Waterfall(
            name='SHAP',
            orientation='v',
            measure=measures,
            x=labels,
            y=values,
            connector={'line': {'color': 'rgb(100, 100, 100)', 'width': 2}},
            marker={'color': colors},
            text=[f"{v:+.3f}" if i > 0 and i < len(values) - 1 else f"{v:.3f}" 
                  for i, v in enumerate(values)],
            textposition='outside',
            textfont={'size': 10}
        ))
        
        fig.update_layout(
            title=f'Explicação Local: Predição = {predicted_class}<br>Amostra #{sample_idx}',
            xaxis_title='Features (ordenadas por importância)',
            yaxis_title='Contribuição para a Predição',
            height=600,
            template='plotly_dark',
            showlegend=False,
            xaxis={'tickangle': -45}
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar force plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return go.Figure()


def get_shap_summary_stats(shap_values, feature_names: List[str]) -> Dict[str, Any]:
    """
    Calcula estatísticas sumárias dos SHAP values.
    
    Args:
        shap_values: SHAP values
        feature_names: Lista com nomes das features
        
    Returns:
        Dict com estatísticas
    """
    try:
        # Calcular importância média
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            if len(shap_values.shape) == 3:
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Top 5 features
        top5_indices = np.argsort(mean_abs_shap)[-5:][::-1]
        top5_features = [(feature_names[i], float(mean_abs_shap[i])) for i in top5_indices]
        
        return {
            'n_features': len(feature_names),
            'top5_features': top5_features,
            'mean_importance': float(mean_abs_shap.mean()),
            'max_importance': float(mean_abs_shap.max()),
            'min_importance': float(mean_abs_shap.min())
        }
        
    except Exception as e:
        logger.error(f"Erro ao calcular summary stats: {e}")
        return {}
