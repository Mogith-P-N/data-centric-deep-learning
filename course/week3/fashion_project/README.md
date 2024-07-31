# Week 3 Project: Active learning on fashion classifier

```
pip install -e .
```

This project will investigate various strategies to identify elements for relabeling.



random results in production 
{'acc': 0.3143000304698944, 'loss': 1.804521918296814}

uncertainty results in production
{'acc': 0.2890999913215637, 'loss': 2.478508472442627}

margin results 
{'acc': 0.4009000360965729, 'loss': 1.7536089420318604}

Entropy results 
{'acc': 0.3011000454425812, 'loss': 7.092942714691162}

augmented results
{'acc': 0.5837000608444214, 'loss': 1.3688806295394897}

Model (random) : 31.3%
Model (Uncertainty) : 28.9%
Model (margin) : 40%
Model (Entropy) : 30.1%
Model (Entropy) : 58.3%

