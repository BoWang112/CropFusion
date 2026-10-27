# -*- coding: utf-8 -*-
"""
baselines_config.py —— 关系抽取基线统一注册表
================================================
本文件只登记真实实现/真实预测文件路径，不构造模拟模型。

仓库内当前可用的真实实现：
  - roberta_copet_casrel : roberta_copet_casrel_merged811.py
                           基础模型 chinese-roberta-wwm-ext/
                           训练权重 checkpoint_merged811/best_model.pt
                           真实预测 checkpoint_merged811/bert_casrel_pred_0.7.json

以下原始基线实现未随仓库提供，必须通过环境变量指向其原始代码/预测文件，
否则评价入口会明确报错而不是伪造结果：
  - CopyRE      : CDPKG_COPYRE_IMPL / CDPKG_COPYRE_PRED
  - SpERT       : CDPKG_SPERT_IMPL / CDPKG_SPERT_PRED
  - TPLinker    : CDPKG_TPLINKER_IMPL / CDPKG_TPLINKER_PRED
  - OneRel      : CDPKG_ONEREL_IMPL / CDPKG_ONEREL_PRED
  - CasRel      : CDPKG_CASREL_IMPL / CDPKG_CASREL_PRED
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, 'chinese-roberta-wwm-ext')
DEFAULT_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoint_merged811')

BASELINES = {
    'roberta_copet_casrel': {
        'display': 'RoBERTa-CoPET-CasRel',
        'status': 'available',
        'implementation': os.path.join(
            BASE_DIR, 'roberta_copet_casrel_merged811.py'),
        # 基础预训练模型（build_transformer_model 的 config/checkpoint）
        'base_model_dir': DEFAULT_MODEL_DIR,
        'base_config': os.path.join(DEFAULT_MODEL_DIR, 'config.json'),
        'base_checkpoint': os.path.join(
            DEFAULT_MODEL_DIR, 'pytorch_model.bin'),
        'vocab_path': os.path.join(DEFAULT_MODEL_DIR, 'vocab.txt'),
        # 已训练的关系抽取权重
        'trained_checkpoint': os.path.join(
            DEFAULT_CHECKPOINT_DIR, 'best_model.pt'),
        # 真实预测文件（JSON 流式对象，每行一个样本）
        'default_pred': os.path.join(
            DEFAULT_CHECKPOINT_DIR, 'bert_casrel_pred_0.7.json'),
        'pred_format': 'json_stream',
        'config': {
            'model_dir': DEFAULT_MODEL_DIR,
            'maxlen': 512,
            'batch_size': 8,
            'entity_types': 16,
            'predicates': 9,
        },
        'notes': '仓库内唯一随附的真实 RE 实现（CoPET-CasRel 结构）与真实预测文件。',
    },
    'copyre': {
        'display': 'CopyRE',
        'status': 'external_required',
        'implementation_env': 'CDPKG_COPYRE_IMPL',
        'pred_env': 'CDPKG_COPYRE_PRED',
        'notes': '原始 CopyRE 实现未随仓库提供，需外部代码/预测文件。',
    },
    'spert': {
        'display': 'SpERT',
        'status': 'external_required',
        'implementation_env': 'CDPKG_SPERT_IMPL',
        'pred_env': 'CDPKG_SPERT_PRED',
        'notes': '原始 SpERT 实现未随仓库提供，需外部代码/预测文件。',
    },
    'tplinker': {
        'display': 'TPLinker',
        'status': 'external_required',
        'implementation_env': 'CDPKG_TPLINKER_IMPL',
        'pred_env': 'CDPKG_TPLINKER_PRED',
        'notes': '原始 TPLinker 实现未随仓库提供，需外部代码/预测文件。',
    },
    'onerel': {
        'display': 'OneRel',
        'status': 'external_required',
        'implementation_env': 'CDPKG_ONEREL_IMPL',
        'pred_env': 'CDPKG_ONEREL_PRED',
        'notes': '原始 OneRel 实现未随仓库提供，需外部代码/预测文件。',
    },
    'casrel': {
        'display': 'CasRel',
        'status': 'external_required',
        'implementation_env': 'CDPKG_CASREL_IMPL',
        'pred_env': 'CDPKG_CASREL_PRED',
        'notes': '纯 CasRel 原始实现未随仓库提供；现有文件是 CoPET-CasRel，'
                 '不能冒充纯 CasRel 基线。',
    },
}


def get_baseline(name):
    if name not in BASELINES:
        raise KeyError(
            f'未知关系抽取基线：{name!r}。可选：{", ".join(sorted(BASELINES))}')
    return dict(BASELINES[name])


def resolve_implementation(name):
    """返回基线的真实实现文件路径；不存在时明确报错。"""
    spec = get_baseline(name)
    if spec['status'] == 'available':
        path = spec.get('implementation', '')
    else:
        path = os.environ.get(spec.get('implementation_env', ''), '')
    if not path or not os.path.exists(path):
        env_name = spec.get('implementation_env', '')
        raise FileNotFoundError(
            f'{spec["display"]} 原始实现未找到。'
            + (f' 请设置 {env_name} 指向原始代码。' if env_name else ''))
    return path


def resolve_prediction(name, pred_path=None):
    """返回基线的真实预测文件路径；未配置时返回空字符串。

    优先级：显式 pred_path > 环境变量 > 仓库内默认预测文件。
    """
    if pred_path:
        return pred_path
    spec = get_baseline(name)
    env_name = spec.get('pred_env', '')
    if env_name:
        path = os.environ.get(env_name, '')
        if path:
            return path
    return spec.get('default_pred', '')


def resolve_checkpoint(name):
    """返回已训练权重路径（若该基线在仓库内提供了训练权重）。"""
    spec = get_baseline(name)
    return spec.get('trained_checkpoint', '')


def describe_baseline(name):
    """返回基线的实现/预测文件与存在性，供 --check / --list-baselines 使用。"""
    spec = get_baseline(name)
    impl = spec.get('implementation', '')
    if spec['status'] != 'available' and spec.get('implementation_env'):
        impl = os.environ.get(spec['implementation_env'], '')
    pred = resolve_prediction(name)
    checkpoint = resolve_checkpoint(name)
    missing = []
    if not impl or not os.path.exists(impl):
        missing.append('implementation')
    if not pred or not os.path.exists(pred):
        missing.append('prediction')
    if checkpoint and not os.path.exists(checkpoint):
        missing.append('trained_checkpoint')
    return {
        'name': name,
        'display': spec['display'],
        'status': spec['status'],
        'implementation': impl,
        'implementation_exists': bool(impl and os.path.exists(impl)),
        'prediction': pred,
        'prediction_exists': bool(pred and os.path.exists(pred)),
        'trained_checkpoint': checkpoint,
        'checkpoint_exists': bool(checkpoint and os.path.exists(checkpoint)),
        'missing': missing,
        'notes': spec.get('notes', ''),
    }


def available_baselines():
    """返回当前具备真实实现或真实预测文件的基线名称。"""
    return [name for name in sorted(BASELINES)
            if describe_baseline(name)['implementation_exists']
            or describe_baseline(name)['prediction_exists']]
