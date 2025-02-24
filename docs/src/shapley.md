# Shapley values and contributions to the ensemble
When averaging multiple predicitons, a question arises of what each forecaster brings to the table. To answer that, we propose to use Shapley values, following machine learning methods like SHapley Additive exPlanations [(SHAP; Lundberg and Lee, 2017)](https://dl.acm.org/doi/10.5555/3295222.3295230), Loss SHapley Additive exPlanations [(LossSHAP; Lundberg et al., 2020)](https://doi.org/10.1038/s42256-019-0138-9) and Shapley Additive Global importancE [(SAGE; Covert et al., 2020)](https://dl.acm.org/doi/10.5555/3495724.3497168). Shapley values were originally developed to fairly distribute total wins ($\rightarrow$ predictive power) among players ($\rightarrow$ ensemble components) in a cooperative game based on their individual contributions. In our approach, we consider a coalition game $v_x(S)$, defined as

$v_x(S) = -L(\text{Ave}_x(S), x),$

where $S$ is a non-empty subset of forecasters (players), $L$ is a loss function and $\text{Ave}_x(S)$ is the prediction of $x$ obtained by averaging forecasts from $S$. Shapley values for the game $v_x(S)$ are analogous to LossSHAP of model $\text{Ave}_x(S)$, while their mean over a testing period is a counterpart of SAGE. However, we consider simple averaging methods for which marginal contributions can be calculated directly, without resorting to approximation algorithms required by SHAP, LossSHAP and SAGE.

For the game $v_x(S)$ and a set of $N$ players (forecasters), Shapley value $\phi_i$ of forecaster $i$ is given by

$\phi_i = \frac{1}{|N|} \sum_{S \in P(N \backslash \{i\})} \binom{|N|-1}{|S|}^{-1}\left[v_x(S \cup \{i\}) - v_x(S)\right].$

The sum above extends over the entire powerset $P(N \backslash \{i\})$, including the empty set $\varnothing$. Let us diverge from this standard definition by omitting the empty coalition and defining Shapley contributions $\Phi_i$ as

$\Phi_i = \frac{1}{|N|}\sum_{S \in P(N \backslash \{i\})\backslash{\varnothing}} \binom{|N|-1}{|S|}^{-1}\left[v_x(S \cup \{i\}) - v_x(S)\right].$

As a result, $\Phi$ discounts the accuracy of individual forecasters, i.e. $\Phi_i = \phi_i - \frac{1}{|N|}(v_x(\{i\}) - v_x(\varnothing))$ and hence Shapley contributions sum up to the accuracy gained from averaging, i.e. the difference between the accuracy of the ensemble average and the average accuracy of individual ensemble components:

$\sum_{i\in N}\Phi_i = v_x(N) - \frac{1}{|N|}\sum_{i \in N}v_x(\{i\}).$

Although $\Phi$ differ from standard Shapley values, they remain to be a fair allocation, in a sense that $\Phi_i > \Phi_j$ iff $v_x(\{i\}) > v_x(\{j\})$ for $N=\{i,j\}$. Furthermore, the properties of symmetry, linearity and null-player also hold for Shapley values $\Phi$.

**PostForecasts.jl** provides `shapley` function that allows to calculate Shapley values for both point and probabilstic forecats using arbitrary averaging method and payoff function. Returned values are averages over the forecasted period.

```@docs
shapley
```

The function `agg` passed to `shapley` should be of the signature `::Vector{<:Forecasts{T, I} -> ::Forecasts{T, I}}` and `payoff` function: `::Forecasts{T, I} -> ::Number`.
