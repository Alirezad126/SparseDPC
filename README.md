# Sparse Differentiable Predictive Control (**SparseDPC**)

SparseDPC demonstrates how **sparse dictionary control policies** can be optimized using  
[Differentiable Predictive Control (DPC)](https://www.sciencedirect.com/science/article/pii/S0959152422000981).  
The entire implementation is built on top of  
[**Neuromancer**](https://github.com/PNNL/Neuromancer).

---

## Running the Code

| Option               | What to Do |
|----------------------|------------|
| **Notebook workflow** | Mark the `src/` folder as the **source root** in your IDE or Jupyter so imports like `SparseDPC.src.sindy.library` work without modifying `sys.path`. |
| **Package install**   | From the repo root, run:  
```bash
pip install -e .
```
Or build a wheel and install:
```bash
python -m build
pip install dist/*.whl
```

---

## 📁 Repository Layout

```
SparseDPC/
├── 5_Full_DPC_Fixed/        ← example notebooks
│   ├── TwoTank_*.ipynb
│   └── VanDerPol_*.ipynb
└── src/
    └── SparseDPC/           ← library code
        ├── sindy/           ← sparse-identification utilities
        ├── trainer/         ← trainers for sparse training
        └── utils/           ← modules like integrators and loggers
```

- 🧪 **Notebooks** for the Two-Tank and Van der Pol systems are in `5_Full_DPC_Fixed/`
- ⚙️ **Source code** for SINDy dynamics, training loops, and utilities is in `src/SparseDPC/`

---

Happy experimenting! ✨
