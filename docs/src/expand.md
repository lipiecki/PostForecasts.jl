# Expanding PostForecasts.jl

Below You can find simple guidelines on how to expand the functionalities of PostForecasts.jl by adding a new model

- create a `mymodel.jl` file in the `src/models` directory, all required functions should be included in this file
- define `MyModel` struct and specify its parent type (`UniRegProbModel` for models that only support single regressor, `MultiRegProbModel` otherwise)

For all models add methods:
```julia
getmodel(::Val{:mymodel}, params::Vararg)::MyModel

_train(m::MyModel, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})
```

For `UniRegProbModel`, add methods:
```julia
_predict(m::MyModel, input::Number, prob::AbstractFloat)::Float64

_predict!(m::MyModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})

_predict!(m::MyModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
```

For `MultiRegProbModel` add methods:
```julia
_predict(m::MyModel, input::Number, prob::AbstractFloat)::Float64

_predict(m::MyModel, input::AbstractVector{<:Number}, prob::AbstractFloat)::Float64

_predict!(m::MyModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})

_predict!(m::MyModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})

nregs(m::MyModel)::Int
```

If your model, once defined, is tied to a specific training window, add:
```julia
matchwindow(m::MyModel, window::Integer)::Bool
```

In the above methods, argument validation is not required (apart from `MyModel` constructor).

Following these steps will allow Your model to be used in `point2prob`, `train` and `predict`/`predict!` methods that are equipped with validation of arguments' dimensionality and values. Add model-specific `train`, `predict`/`predict!` methods to `src/train.jl` or `src/predict.jl` only if additional validation or new function signatures are required (see `src/train.jl` and `src/predict.jl`)
