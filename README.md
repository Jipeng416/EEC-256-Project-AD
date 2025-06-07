# EEC‑256‑Project‑AD 🚀  
**Replicating “In‑context Reinforcement Learning with Algorithm Distillation” (ICLR 2023)**  
Course project for *EEC 256 – Reinforcement Learning* @ UC Davis  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zyUr75V9fyaVqgBbO2mVyN9Ctb9RnVh2#scrollTo=g9fCtI-9FQx1)

---

## 1 · Installation 🛠️
Tested on **Python 3.10 + CUDA 11.x**.

```bash
git clone https://github.com/Jipeng416/EEC-256-Project-AD.git
cd EEC-256-Project-AD

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements/requirements.txt
export PYTHONPATH=.             # make local src/ importable
```

> **Just want a demo?** click the Colab badge above.

---

## 2 · Repo map
```text
.
├── notebooks/                 # demos & analysis
├── scripts/                   # stage‑1/2/3 helpers
├── src/                       # env, PPO, DT, eval
├── saved_data/                # goals + learning_history + models
└── static/                    # figures for report
```

---

## 3 · Quick start 🏃‍♀️
### 3.1 Generate goals
```bash
python src/data/generate_goals.py
```
### 3.2 Collect PPO histories (Stage 1)
```bash
bash scripts/collect_data.sh
```
Have zipped files?
```bash
unzip -o 'saved_data/learning history/ppo-*.zip' -d saved_data/learning_history
python utils/rename_histories.py
```
### 3.3 Train AD (Stage 2)
```bash
python src/dt/train.py
# reward head
python src/dt/train.py --config.add-reward_head
```
### 3.4 Evaluate (Stage 3)
```bash
python src/check_in_context/eval_one_model.py   --model-dir saved_data/saved_models/classical_model   --permutations-file saved_data/permutations_9.txt   --n-repeats 5 --eval-episodes 100 --no-only-last
```

---

## 4 · Core hyper‑parameters
| Category | Setting | Value |
|----------|---------|-------|
| **PPO**  | LR / optimiser | 3e‑4 / Adam |
|          | γ / λ          | 0.99 / 0.95 |
|          | clip ε         | 0.2 |
|          | rollout        | 128 × 8 envs |
|          | steps / task   | 200 k |
| **AD‑DT**| layers / hidden| 4 / 64 |
|          | heads / FF dim | 4 / 256 |
|          | seq len        | 60 |
|          | optimiser      | AdamW 1e‑4 |

---

## 5 · Colab one‑liner ☁️
```bash
!git clone https://github.com/Jipeng416/EEC-256-Project-AD && pip install -q -r EEC-256-Project-AD/requirements/requirements_colab.txt && python EEC-256-Project-AD/scripts/train_ad.sh --quick
```

---

## 6 · Results
See all curves and logs on wandb: <https://wandb.ai/your_user/eec256-ad>

---

## 7 · Acknowledgements
* Original AD idea & figure: DeepMind  
* PPO adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl)  
* Decision‑Transformer base: CORL team

---

## 8 · Licence
MIT – see `LICENSE`.
