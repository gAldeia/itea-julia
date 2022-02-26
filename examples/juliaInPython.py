# Deve ter a biblioteca julia no python, e a PyCall em julia

from julia.api import Julia
jl = Julia(compiled_modules=False)

# Importando o escopo global onde tudo é carregado em julia
from julia import Main

jl.using("StatsBase")
jl.using("ITEAregressor") # Deve ter a biblioteca localmente instalada com Julia

import pandas as pd

df = pd.read_csv('../experiments/datasets/airfoil.csv')

X, y = df.iloc[:, :-1], df.iloc[:, -1]

Main.ITEA(
    X,
    y,
    [Main.identity, Main.sin, Main.cos, Main.tan, Main.sqrt, Main.log, Main.exp, Main.exp10],
    max(1, X.shape[1]-2),

    # named arguments (optional)
    tourn_size = 2,
    expo_bounds = (-2, 2),
    terms_bounds=(1, 5),
    popsize =100,
    gens = 100,
    mutationWeights = Main.ProbabilityWeights([.11, .11, .11, .33, .33]),
    adjust_method ="levenberg_marquardt_adj",
)