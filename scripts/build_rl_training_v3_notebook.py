"""Generate notebooks/rl_training_v3_notebook.ipynb — code cells only, no comments."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "rl_training_v3_notebook.ipynb"


def _cell(source: str, cell_id: str) -> dict:
    lines = source.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


CELLS = [
    _cell(
        """
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
assert os.path.exists('/content'), 'Colab required'
subprocess.run(['nvidia-smi'], check=True)

REPO_URL = 'https://github.com/prashere/evaluator_gym.git'
try:
    from google.colab import userdata
    github_token = userdata.get('GITHUB_TOKEN')
except Exception:
    github_token = os.environ.get('GITHUB_TOKEN')
if github_token:
    REPO_URL = f'https://{github_token}@github.com/prashere/evaluator_gym.git'

REPO = Path('/content/evaluator_gym')
COLAB_BRANCH = os.environ.get('EVALUATOR_GYM_COLAB_BRANCH', 'rl_v3')
if not REPO.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO)], check=True)
os.chdir(REPO)
subprocess.run(['git', 'fetch', 'origin', COLAB_BRANCH], check=True)
subprocess.run(['git', 'checkout', COLAB_BRANCH], check=True)
subprocess.run(['git', 'pull', '--ff-only', 'origin', COLAB_BRANCH], check=True)

_bootstrap_file = REPO / 'src/evaluator_gym/training/colab_bootstrap.py'
_spec = importlib.util.spec_from_file_location('colab_bootstrap', _bootstrap_file)
_bootstrap = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bootstrap)
SRC = _bootstrap.install_repo_src(REPO)
print(SRC)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(REPO)], check=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'transformers==4.57.6', 'peft==0.17.1', 'bitsandbytes==0.47.0',
    'accelerate==1.10.1', 'mlflow==3.10.0', 'matplotlib>=3.8,<4', 'tqdm>=4.66,<5',
], check=True)
_bootstrap.install_repo_src(REPO)
print(_bootstrap.verify_imports_v3())

from google.colab import drive
drive.mount('/content/drive')

from evaluator_gym.training.phase07_core import ensure_output_root
from evaluator_gym.training.phase07v3_core import DEFAULT_OUTPUT_ROOT_V3
from evaluator_gym.training.phase07v3_notebook import verify_bitsandbytes, write_run_manifest

print('bitsandbytes', verify_bitsandbytes())
OUTPUT_ROOT = ensure_output_root(DEFAULT_OUTPUT_ROOT_V3)
COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
write_run_manifest(OUTPUT_ROOT, repo_commit=COMMIT)
print('OUTPUT_ROOT', OUTPUT_ROOT)
print('COMMIT', COMMIT)
""",
        "c1",
    ),
    _cell(
        """
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

from evaluator_gym.training.phase07v3_core import build_phase07v3_splits

TRAIN_CONFIG, HELDOUT_POOL_CONFIG, TRAIN_TASK_ROWS, HELDOUT_TASK_ROWS = build_phase07v3_splits()
TRAIN_TASKS = [row.as_dict() for row in TRAIN_TASK_ROWS]
HELDOUT_TASKS = [row.as_dict() for row in HELDOUT_TASK_ROWS]
print(TRAIN_CONFIG.to_dict())
print(HELDOUT_POOL_CONFIG.to_dict())
print(dict(Counter(row['tier'] for row in TRAIN_TASKS)))
print([row['task_id'] for row in HELDOUT_TASKS])
""",
        "c2",
    ),
    _cell(
        """
import json
import random
import sys
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

import numpy as np
import torch
from evaluator_gym.rubric import RUBRIC_VERSION
from evaluator_gym.training_rubric import TRAINING_RUBRIC_VERSION
from evaluator_gym.training.phase07_core import RUN_SEED, ensure_output_root
from evaluator_gym.training.phase07v3_core import (
    BETAS,
    DEFAULT_OUTPUT_ROOT_V3,
    GROUP_SIZE,
    HELDOUT_ROLLOUTS,
    MAX_COMPLETION_TOKENS,
    MAX_RESAMPLE_ATTEMPTS,
    MAX_RETRIES,
    MAX_TOTAL_COMPLETIONS,
    MIN_MODEL_MAX_POSITION,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SELECTION_NOTE,
    PEAK_STEP_GIB,
    PREFLIGHT_PROBES_PER_TASK,
    RL_LEARNING_RATE,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    SMOKE_MAX_RESAMPLE_ATTEMPTS,
    SMOKE_STEPS,
    TARGET_OPTIMIZER_STEPS,
    TRAIN_STEPS,
    build_training_schedule_v3,
    evaluate_sft_gate,
    summarize_dual_evaluation_v3,
    summarize_preflight_v3,
    validate_model_max_position,
    validate_trainable_pool_v3,
)
from evaluator_gym.training.phase07v3_notebook import (
    assert_gpu_headroom,
    assert_peak_within_budget,
    backward_rloo_policy_step,
    build_8bit_optimizer,
    load_sft_adapter_weights,
    log_gpu_memory,
    release_gpu_memory,
    sft_adapter_dir,
    sft_adapter_ready,
    sft_completion_loss,
    token_statistics,
)
from evaluator_gym.training.phase07v3_runtime import (
    append_jsonl,
    exploit_search_v3,
    read_jsonl,
    run_async,
    score_text_dual_v3,
)
from evaluator_gym.training.phase07_live import LiveRunLogger, tqdm_progress
from evaluator_gym.training.sft_reference import build_sft_examples

OUTPUT_ROOT = ensure_output_root(globals().get('OUTPUT_ROOT', DEFAULT_OUTPUT_ROOT_V3))
assert RUBRIC_VERSION == '0.1.2', f'Expected eval rubric 0.1.2, got {RUBRIC_VERSION}'
assert TRAINING_RUBRIC_VERSION == 'train-0.1.0'
assert torch.cuda.is_available(), 'Colab GPU required'
print(torch.cuda.get_device_name(0))
print(MODEL_ID, MODEL_REVISION)
print(MODEL_SELECTION_NOTE)
print(OUTPUT_ROOT)
print(GROUP_SIZE, MAX_COMPLETION_TOKENS, HELDOUT_ROLLOUTS)
""",
        "c3",
    ),
    _cell(
        """
import sys
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from evaluator_gym.training.phase07v3_core import (
    GROUP_SIZE,
    MAX_COMPLETION_TOKENS,
    MAX_RESAMPLE_ATTEMPTS,
    MAX_RETRIES,
    compute_rloo_advantages,
    is_mixed_group,
)


def build_policy():
    set_seed(RUN_SEED)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=quantization,
        device_map={'': 0},
        torch_dtype=torch.float16,
        attn_implementation='sdpa',
    )
    validate_model_max_position(int(model.config.max_position_embeddings))
    global POLICY_MAX_POSITION
    POLICY_MAX_POSITION = int(model.config.max_position_embeddings)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    ))
    unexpected = [name for name, p in model.named_parameters() if p.requires_grad and 'lora_' not in name]
    assert not unexpected, f'Frozen-reference invariant failed: {unexpected[:5]}'
    return model, tokenizer


def render_prompt(task, tokenizer):
    text = tokenizer.apply_chat_template(task['prompt'], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors='pt').to('cuda')
    assert encoded['input_ids'].shape[1] + MAX_COMPLETION_TOKENS <= POLICY_MAX_POSITION
    return encoded


def generate_one(model, tokenizer, encoded, prompt_length, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=0.85,
            max_new_tokens=MAX_COMPLETION_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output[0, prompt_length:].detach().clone()
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_ids, text


def generate_valid_group(model, tokenizer, task, nominal_step, rejection_path, live_logger=None):
    encoded = render_prompt(task, tokenizer)
    prompt_length = encoded['input_ids'].shape[1]
    samples = []
    model.eval()
    slot_iter = tqdm_progress(range(GROUP_SIZE), desc=f'group {task[\"task_id\"]} s{nominal_step}', total=GROUP_SIZE, leave=False)
    for group_index in slot_iter:
        accepted = None
        for retry in range(MAX_RETRIES):
            sample_seed = RUN_SEED + nominal_step * 1000 + group_index * 10 + retry
            completion_ids, text = generate_one(model, tokenizer, encoded, prompt_length, sample_seed)
            training_reward, eval_reward, parse_result, audit = run_async(score_text_dual_v3(task, text))
            if live_logger is not None:
                live_logger.log_group_slot(
                    nominal_step=nominal_step,
                    task_id=task['task_id'],
                    group_index=group_index,
                    retry=retry,
                    scored=training_reward is not None,
                    reward=training_reward,
                    error_class=parse_result.get('error_class'),
                )
            candidate = {
                'task_id': task['task_id'], 'tier': task['tier'], 'nominal_step': nominal_step,
                'group_index': group_index, 'retry': retry, 'seed': sample_seed,
                'completion': text, 'completion_tokens': int(completion_ids.numel()),
                'reward': training_reward, 'training_reward': training_reward,
                'eval_reward': eval_reward, 'parse_result': parse_result, 'reward_audit': audit,
            }
            if training_reward is None:
                append_jsonl(rejection_path, candidate)
                continue
            candidate['prompt_ids'] = encoded['input_ids'][0].detach().clone()
            candidate['completion_ids'] = completion_ids.detach().clone()
            accepted = candidate
            break
        if accepted is None:
            model.eval()
            return None
        samples.append(accepted)
    return samples


def generate_mixed_group(model, tokenizer, task, nominal_step, rejection_path, max_resample_attempts, live_logger=None):
    completions_used = 0
    last = None
    for attempt in range(max_resample_attempts):
        samples = generate_valid_group(
            model, tokenizer, task, nominal_step + attempt * 10000, rejection_path, live_logger=live_logger,
        )
        completions_used += GROUP_SIZE
        if samples is None:
            continue
        last = samples
        rewards = [float(sample['training_reward']) for sample in samples]
        if is_mixed_group(rewards):
            return samples, attempt + 1, completions_used
        for sample in samples:
            sample.pop('prompt_ids', None)
            sample.pop('completion_ids', None)
    return last, max_resample_attempts, completions_used


def evaluate_policy(model, tokenizer, run_name):
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / 'heldout_rollouts.jsonl'
    summary_path = run_dir / 'heldout_summary.json'
    if summary_path.is_file() and transcript_path.is_file():
        return json.loads(summary_path.read_text())
    if transcript_path.exists():
        transcript_path.unlink()
    live = LiveRunLogger(run_name, run_dir)
    live.print_banner(f'held-out {run_name}')
    model.eval()
    total = len(HELDOUT_TASKS) * HELDOUT_ROLLOUTS
    rollout_index_global = 0
    progress = tqdm_progress(range(total), desc=f'held-out {run_name}', total=total)
    for task_index, task in enumerate(HELDOUT_TASKS):
        encoded = render_prompt(task, tokenizer)
        prompt_length = encoded['input_ids'].shape[1]
        for rollout_index in range(HELDOUT_ROLLOUTS):
            seed = RUN_SEED + task_index * HELDOUT_ROLLOUTS + rollout_index
            ids, text = generate_one(model, tokenizer, encoded, prompt_length, seed)
            training_reward, eval_reward, parse_result, audit = run_async(score_text_dual_v3(task, text))
            rollout_index_global += 1
            append_jsonl(transcript_path, {
                'run': run_name, 'task_id': task['task_id'], 'tier': task['tier'],
                'rollout_index': rollout_index, 'seed': seed, 'completion': text,
                'completion_tokens': int(ids.numel()), 'reward': eval_reward,
                'eval_reward': eval_reward, 'training_reward': training_reward,
                'parse_result': parse_result, 'reward_audit': audit,
            })
            live.log_eval_rollout(
                index=rollout_index_global, total=total, task_id=task['task_id'],
                tier=task['tier'], reward=eval_reward, completion_tokens=int(ids.numel()),
                parse_result=parse_result,
            )
            progress.update(1)
    progress.close()
    rows = read_jsonl(transcript_path)
    summary = summarize_dual_evaluation_v3(rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    live.emit(json.dumps(summary))
    return summary
""",
        "c4",
    ),
    _cell(
        """
import json
import sys
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

base_summary = OUTPUT_ROOT / 'base' / 'heldout_summary.json'
if base_summary.is_file():
    BASE_HELDOUT = json.loads(base_summary.read_text())
else:
    log_gpu_memory('pre-base')
    assert_gpu_headroom(label='base')
    base_model, base_tokenizer = build_policy()
    BASE_HELDOUT = evaluate_policy(base_model, base_tokenizer, 'base')
    del base_model
    del base_tokenizer
    release_gpu_memory(globals())
print(json.dumps(BASE_HELDOUT, indent=2))
""",
        "c5",
    ),
    _cell(
        """
import json
import sys
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

from evaluator_gym.training.sft_reference import build_sft_examples

SFT_EXAMPLES = build_sft_examples(TRAIN_TASK_ROWS)
adapter_path = sft_adapter_dir(OUTPUT_ROOT)
if sft_adapter_ready(OUTPUT_ROOT):
    print(json.dumps({'phase': 'sft_ready', 'adapter': str(adapter_path)}))
else:
    log_gpu_memory('pre-sft')
    assert_gpu_headroom(label='sft')
    sft_model, sft_tokenizer = build_policy()
    optimizer = build_8bit_optimizer(sft_model, SFT_LEARNING_RATE)
    sft_model.train()
    for epoch in range(SFT_EPOCHS):
        for example in tqdm_progress(SFT_EXAMPLES, desc=f'sft {epoch + 1}'):
            encoded = render_prompt({'prompt': example['prompt']}, sft_tokenizer)
            completion_ids = sft_tokenizer(
                example['completion'], return_tensors='pt', add_special_tokens=False,
            )['input_ids'][0].to('cuda')
            if int(completion_ids.numel()) > MAX_COMPLETION_TOKENS:
                completion_ids = completion_ids[:MAX_COMPLETION_TOKENS]
            loss = sft_completion_loss(sft_model, encoded['input_ids'][0], completion_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss, completion_ids, encoded
    adapter_path.mkdir(parents=True, exist_ok=True)
    sft_model.save_pretrained(adapter_path)
    sft_tokenizer.save_pretrained(adapter_path)
    del sft_model
    del sft_tokenizer
    release_gpu_memory(globals())
    print(json.dumps({'phase': 'sft_complete', 'next': 'restart_runtime', 'adapter': str(adapter_path)}))
    raise SystemExit('SFT_ADAPTER_SAVED')
""",
        "c6",
    ),
    _cell(
        """
import json
import sys
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

assert sft_adapter_ready(OUTPUT_ROOT), str(sft_adapter_dir(OUTPUT_ROOT))
log_gpu_memory('pre-post-sft')
assert_gpu_headroom(label='post-sft')
policy_model, policy_tokenizer = build_policy()
load_sft_adapter_weights(policy_model, sft_adapter_dir(OUTPUT_ROOT))
POST_SFT_HELDOUT = evaluate_policy(policy_model, policy_tokenizer, 'post_sft')
GATE = evaluate_sft_gate(POST_SFT_HELDOUT)
(OUTPUT_ROOT / 'gate_result.json').write_text(json.dumps(GATE, indent=2))
print(json.dumps(POST_SFT_HELDOUT, indent=2))
print(json.dumps(GATE, indent=2))
if not GATE['passed']:
    del policy_model
    del policy_tokenizer
    release_gpu_memory(globals())
    raise SystemExit('SFT_GATE_FAILED')
del policy_model
del policy_tokenizer
release_gpu_memory(globals())
""",
        "c7",
    ),
    _cell(
        """
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

preflight_path = OUTPUT_ROOT / 'preflight.json'
if preflight_path.is_file():
    PREFLIGHT = json.loads(preflight_path.read_text())
else:
    log_gpu_memory('pre-preflight')
    assert_gpu_headroom(label='preflight')
    preflight_model, preflight_tokenizer = build_policy()
    load_sft_adapter_weights(preflight_model, sft_adapter_dir(OUTPUT_ROOT))
    probes = []
    live = LiveRunLogger('preflight', OUTPUT_ROOT)
    preflight_model.eval()
    for task_index, task_row in tqdm_progress(list(enumerate(TRAIN_TASK_ROWS)), desc='preflight'):
        task = task_row.as_dict()
        encoded = render_prompt(task, preflight_tokenizer)
        prompt_length = encoded['input_ids'].shape[1]
        for probe_index in range(PREFLIGHT_PROBES_PER_TASK):
            seed = RUN_SEED + 500000 + task_index * 100 + probe_index
            ids, text = generate_one(preflight_model, preflight_tokenizer, encoded, prompt_length, seed)
            training_reward, eval_reward, parse_result, _ = run_async(score_text_dual_v3(task, text))
            probes.append({
                'task_id': task['task_id'], 'tier': task['tier'], 'probe_index': probe_index,
                'reward': training_reward, 'training_reward': training_reward,
                'eval_reward': eval_reward, 'parse_result': parse_result,
                'completion_tokens': int(ids.numel()),
            })
    PREFLIGHT = summarize_preflight_v3(probes)
    validate_trainable_pool_v3(PREFLIGHT)
    preflight_path.write_text(json.dumps(PREFLIGHT, indent=2))
    del preflight_model
    del preflight_tokenizer
    release_gpu_memory(globals())

TRAINABLE_IDS = set(PREFLIGHT['trainable_task_ids'])
TRAINING_SCHEDULE = build_training_schedule_v3(TRAINABLE_IDS, TRAIN_TASK_ROWS, TRAIN_STEPS)
print(PREFLIGHT['trainable_by_tier'], len(TRAINABLE_IDS))
print([row.task_id if row else None for row in TRAINING_SCHEDULE])
""",
        "c8",
    ),
    _cell(
        """
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

from peft import get_peft_model_state_dict, set_peft_model_state_dict

from evaluator_gym.training.phase07_core import build_training_metric
from evaluator_gym.training.phase07v3_core import decide_training_step_v3


def save_checkpoint(run_dir, nominal_step, model, optimizer, optimizer_applied_steps):
    checkpoint = run_dir / f'checkpoint-{nominal_step}'
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save({
        'nominal_step': nominal_step,
        'optimizer_applied_steps': optimizer_applied_steps,
        'adapter': get_peft_model_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(),
    }, checkpoint / 'state.pt')
    return checkpoint


def train_run(beta, run_name, total_steps, trainable_ids, training_schedule, resume_checkpoint=None, max_resample_attempts=MAX_RESAMPLE_ATTEMPTS):
    assert beta > 0
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.jsonl'
    rollout_path = run_dir / 'training_rollouts.jsonl'
    rejection_path = run_dir / 'rejected_unscored.jsonl'
    run_config = {
        'model': MODEL_ID,
        'model_revision': MODEL_REVISION,
        'training_rubric_version': TRAINING_RUBRIC_VERSION,
        'eval_rubric_version': RUBRIC_VERSION,
        'beta': beta,
        'seed': RUN_SEED,
        'total_steps': total_steps,
        'group_size': GROUP_SIZE,
        'max_retries': MAX_RETRIES,
        'max_completion_tokens': MAX_COMPLETION_TOKENS,
        'max_resample_attempts': max_resample_attempts,
        'advantage_estimator': 'rloo',
        'selection_method': 'unique_reward_qualified_v3',
        'trainable_task_ids': sorted(trainable_ids),
        'schedule_task_ids': [row.task_id if row else None for row in training_schedule[:total_steps]],
        'train_generator': TRAIN_CONFIG.to_dict(),
        'heldout_pool_generator': HELDOUT_POOL_CONFIG.to_dict(),
    }
    log_gpu_memory(f'pre-{run_name}')
    assert_gpu_headroom(label=run_name)
    model, tokenizer = build_policy()
    load_sft_adapter_weights(model, sft_adapter_dir(OUTPUT_ROOT))
    optimizer = build_8bit_optimizer(model, RL_LEARNING_RATE)
    start_step = 0
    optimizer_applied_steps = 0
    total_completions = 0
    if resume_checkpoint:
        state = torch.load(Path(resume_checkpoint) / 'state.pt', map_location='cpu', weights_only=False)
        set_peft_model_state_dict(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step = state['nominal_step']
        optimizer_applied_steps = state.get('optimizer_applied_steps', 0)
    elif metrics_path.exists() or rollout_path.exists():
        raise RuntimeError(f'{run_dir} already contains a run')
    (run_dir / 'config.json').write_text(json.dumps(run_config, indent=2))
    skip_counts = Counter()
    live = LiveRunLogger(run_name, run_dir)
    live.print_banner(f'{run_name} beta={beta:g}')
    progress = tqdm_progress(range(start_step, total_steps), desc=run_name, total=total_steps, initial=start_step)
    for step in progress:
        if total_completions >= MAX_TOTAL_COMPLETIONS:
            break
        nominal_step = step + 1
        task_row = training_schedule[step] if step < len(training_schedule) else None
        if task_row is None:
            metric = build_training_metric(
                nominal_step=nominal_step, task=None, rewards=None, kl=None, entropy=None,
                mean_completion_length=None, optimizer_applied=False, skip_reason='no_trainable_task',
            )
            append_jsonl(metrics_path, metric)
            skip_counts['no_trainable_task'] += 1
            live.log_training_step(metric, beta=beta)
            continue
        task = task_row.as_dict()
        samples, resample_attempts, used = generate_mixed_group(
            model, tokenizer, task, nominal_step, rejection_path, max_resample_attempts, live_logger=live,
        )
        total_completions += used
        if samples is None:
            metric = build_training_metric(
                nominal_step=nominal_step, task=task_row, rewards=None, kl=None, entropy=None,
                mean_completion_length=None, optimizer_applied=False, skip_reason='incomplete_group',
            )
            metric['resample_attempts'] = resample_attempts
            append_jsonl(metrics_path, metric)
            skip_counts['incomplete_group'] += 1
            live.log_training_step(metric, beta=beta)
            continue
        reward_values = [float(sample['training_reward']) for sample in samples]
        optimizer_applied, skip_reason = decide_training_step_v3(
            rewards=reward_values, group_complete=True, resample_accepted=is_mixed_group(reward_values),
        )
        mean_length = fmean(sample['completion_tokens'] for sample in samples)
        if not optimizer_applied:
            skip_counts[skip_reason or 'unknown'] += 1
            metric = build_training_metric(
                nominal_step=nominal_step, task=task_row, rewards=reward_values,
                kl=None, entropy=None, mean_completion_length=mean_length,
                optimizer_applied=False, skip_reason=skip_reason, group_rewards=reward_values,
            )
            metric['resample_attempts'] = resample_attempts
            metric['advantage_estimator'] = 'rloo'
            append_jsonl(metrics_path, metric)
            for sample in samples:
                append_jsonl(rollout_path, {k: v for k, v in sample.items() if k not in ('prompt_ids', 'completion_ids')})
            live.log_training_step(metric, beta=beta)
            continue
        advantages = torch.tensor(compute_rloo_advantages(reward_values), device='cuda')
        stats = backward_rloo_policy_step(
            optimizer,
            model,
            advantages=advantages,
            samples=samples,
            beta=beta,
            token_statistics_fn=token_statistics,
        )
        optimizer_applied_steps += 1
        assert_peak_within_budget(label=run_name)
        metric = build_training_metric(
            nominal_step=nominal_step, task=task_row, rewards=reward_values,
            kl=stats['kl'], entropy=stats['entropy'], mean_completion_length=mean_length,
            optimizer_applied=True, skip_reason=None, group_rewards=reward_values,
        )
        metric['loss'] = stats['loss']
        metric['mean_eval_reward'] = fmean(float(sample['eval_reward'] or 0.0) for sample in samples)
        metric['training_rubric_version'] = TRAINING_RUBRIC_VERSION
        metric['eval_rubric_version'] = RUBRIC_VERSION
        metric['resample_attempts'] = resample_attempts
        metric['advantage_estimator'] = 'rloo'
        metric['rloo_advantages'] = compute_rloo_advantages(reward_values)
        append_jsonl(metrics_path, metric)
        for sample in samples:
            append_jsonl(rollout_path, {k: v for k, v in sample.items() if k not in ('prompt_ids', 'completion_ids')})
        live.log_training_step(metric, beta=beta)
        if optimizer_applied_steps % 10 == 0 or nominal_step == total_steps:
            save_checkpoint(run_dir, nominal_step, model, optimizer, optimizer_applied_steps)
    progress.close()
    model.save_pretrained(run_dir / 'final-adapter')
    tokenizer.save_pretrained(run_dir / 'final-adapter')
    print(json.dumps({'run': run_name, 'optimizer_applied': optimizer_applied_steps, 'skip_counts': dict(skip_counts), 'total_completions': total_completions}))
    return model, tokenizer, run_dir


log_gpu_memory('pre-smoke')
assert_gpu_headroom(label='smoke')
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
smoke_model, smoke_tokenizer, smoke_dir = train_run(
    BETAS[0], 'smoke', SMOKE_STEPS, TRAINABLE_IDS, TRAINING_SCHEDULE[:SMOKE_STEPS],
    max_resample_attempts=SMOKE_MAX_RESAMPLE_ATTEMPTS,
)
assert_peak_within_budget(label='smoke')
del smoke_model
del smoke_tokenizer
release_gpu_memory(globals())
smoke_checkpoints = sorted(smoke_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
if smoke_checkpoints:
    smoke_model, smoke_tokenizer, smoke_dir = train_run(
        BETAS[0], 'smoke', SMOKE_STEPS + 1, TRAINABLE_IDS, TRAINING_SCHEDULE[:SMOKE_STEPS + 1],
        smoke_checkpoints[-1], max_resample_attempts=SMOKE_MAX_RESAMPLE_ATTEMPTS,
    )
    del smoke_model
    del smoke_tokenizer
    release_gpu_memory(globals())
print('Smoke and resume passed')
""",
        "c9",
    ),
    _cell(
        """
import json
import sys
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

low_model, low_tokenizer, low_dir = train_run(BETAS[0], 'beta-0.01', TARGET_OPTIMIZER_STEPS, TRAINABLE_IDS, TRAINING_SCHEDULE)
LOW_HELDOUT = evaluate_policy(low_model, low_tokenizer, 'beta-0.01')
del low_model
del low_tokenizer
release_gpu_memory(globals())

strong_model, strong_tokenizer, strong_dir = train_run(BETAS[1], 'beta-0.1', TARGET_OPTIMIZER_STEPS, TRAINABLE_IDS, TRAINING_SCHEDULE)
STRONG_HELDOUT = evaluate_policy(strong_model, strong_tokenizer, 'beta-0.1')
del strong_model
del strong_tokenizer
release_gpu_memory(globals())

COMPARISON = {
    'base': BASE_HELDOUT,
    'post_sft': POST_SFT_HELDOUT,
    'beta-0.01': LOW_HELDOUT,
    'beta-0.1': STRONG_HELDOUT,
    'training_rubric_version': TRAINING_RUBRIC_VERSION,
    'eval_rubric_version': RUBRIC_VERSION,
}
(OUTPUT_ROOT / 'heldout_comparison.json').write_text(json.dumps(COMPARISON, indent=2))
print(json.dumps(COMPARISON, indent=2))
""",
        "c10",
    ),
    _cell(
        """
import json
import sys
from pathlib import Path

REPO = Path('/content/evaluator_gym')
_src = str((REPO / 'src').resolve())
if _src not in sys.path:
    sys.path.insert(0, _src)

from evaluator_gym.training.colab_bootstrap import ensure_colab_repo_path

ensure_colab_repo_path(REPO)

import matplotlib.pyplot as plt

FIGURE_DIR = OUTPUT_ROOT / 'figures'
FIGURE_DIR.mkdir(exist_ok=True)
runs = {
    'beta=0.01': read_jsonl(low_dir / 'metrics.jsonl'),
    'beta=0.1': read_jsonl(strong_dir / 'metrics.jsonl'),
}


def save_curve(field, ylabel, filename):
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, rows in runs.items():
        points = [(row['nominal_step'], row[field]) for row in rows if row.get(field) is not None and row.get('optimizer_applied')]
        axis.plot([x for x, _ in points], [y for _, y in points], marker='o', markersize=3, label=label)
    axis.set(xlabel='Nominal step', ylabel=ylabel, title=ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f'{filename}.png', dpi=140)
    plt.close(figure)


save_curve('mean_reward', 'Mean training reward', 'reward')
save_curve('kl', 'Frozen-reference k3 KL', 'kl')
save_curve('entropy', 'Last-token entropy', 'entropy')
save_curve('mean_completion_length', 'Completion length', 'completion-length')
save_curve('exact_pass_rate', 'Exact pass rate', 'per-tier-pass-rate')

labels = ['base', 'post_sft', 'beta-0.01', 'beta-0.1']
figure, axis = plt.subplots(figsize=(7, 4))
axis.bar(labels, [COMPARISON[name]['mean_reward_scored'] for name in labels])
axis.set(ylabel='Held-out mean eval reward', title='Held-out before/after')
figure.tight_layout()
figure.savefig(FIGURE_DIR / 'heldout-before-after.png', dpi=140)
plt.close(figure)

AUDIT = {'beta-0.01': exploit_search_v3(low_dir), 'beta-0.1': exploit_search_v3(strong_dir)}
(OUTPUT_ROOT / 'exploit_search.json').write_text(json.dumps(AUDIT, indent=2, default=str))
print(json.dumps({key: {k: v for k, v in row.items() if k != 'candidates'} for key, row in AUDIT.items()}, indent=2))
""",
        "c11",
    ),
]


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": CELLS,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
