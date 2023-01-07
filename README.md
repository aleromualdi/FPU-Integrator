# Fermi-Pasta-Ulam integrator based on Verlet algorithm

This repository contains the code that implements an integrator for the Fermi-Pasta-Ulam based on Verlet algorithm

# Installation

The code runs on Python 3.9

To install requirements:

```
pip install -r requirements.txt
```

# Usage
 
 1. To generate a single trajectory, for a given set of parameters, run the script:

```
 python scripts/generate_trajectory.py <args>
```

For example, to run simulation with non-linearliy coefficient beta=0.5, max integration time 20000, and save the results to `output/my_trajectory`, run:

```
python scripts/generate_trajectory.py --output-dir-name "my_trajectory" --beta 0.5 --tmax 20000
```

2. To generate a dataset of trajectories for different beta values

```
python scripts/create_dataset.py 
```

Noe: remember to change the parameters hardcoded in the script `scripts/create_dataset.py`.


3. To run the notebooks

```
jupyter notebook
```


# ToDos

- [ ] generate dataset
  - [x] write script for generation of dataset to run on cluster
  - [ ] generate dataset running the script on cluster

- [ ] investigate non-ergotic -> ergotic biforcation
    - [ ] PCA
        - [ ] create plot of PC explained variance ratio vs beta
    - [ ] persistence homology graphs
        - [ ] find homologies across biforcation
    - [ ] Mapper
        - [ ]  find topological summary in torus (non-ergotic)

- [ ] summarise findings in OverLeaf


