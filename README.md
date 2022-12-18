# Fermi-Pasta-Ulam integrator based on Verlet algorithm

This repository contains the code that implements an integrator for the Fermi-Pasta-Ulam based on Verlet algorithm

# Installation

The code runs on Python 3.9

To install requirements:

```
pip install -r requirements.txt
```

# Usage
 
 To run the script run

```
 python scripts/script.py <args>
```

For example, to run simulation with non-linearliy coefficient beta=0.5, max integration time 20000, and save the results to `output/prova`, run:

python scripts/script.py --output-dir-name "prova" --beta 0.5 --tmax 20000


To run the notebooks

```
jupyter notebook
```


