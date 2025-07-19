abstract type AbstractCoMetModel end

# COMET Model Implementation
struct COMETModel <: AbstractCoMetModel
    nn::Flux.Chain
    nstates::Int
    ncom::Int
end

Flux.@functor COMETModel

function COMETModel(nstates::Int, ncom::Int; nhidden::Int=250)
    nn = Chain(
        Dense(nstates, nhidden, logsig),
        Dense(nhidden, nhidden, logsig),
        Dense(nhidden, nhidden, logsig),
        Dense(nhidden, nstates + ncom)
    )
    return COMETModel(nn, nstates, ncom)
end

function (m::COMETModel)(states::AbstractArray)
    nn_out = m.nn(states)
    s_dot_0 = nn_out[1:m.nstates, :]
    c = nn_out[m.nstates+1:end, :]

    if m.ncom == 0
        return s_dot_0, s_dot_0, c
    end

    # QR decomposition for orthogonalization [cite: 65]
    ∇c = Zygote.jacobian(s -> m.nn(s)[m.nstates+1:end, :], states)[1]
    A = hcat(∇c, s_dot_0)
    Q, R = qr(A)
    s_dot = Q[:, end] .* R[end, end]
    return s_dot, s_dot_0, c
end

# --- Meta-COMET Model Implementation ---
struct SVD_FC
    S::AbstractMatrix
    V::AbstractMatrix
    D::AbstractMatrix
end
Flux.@functor SVD_FC

(l::SVD_FC)(h) = l.S * (relu.(l.V) * (l.D' * h))

struct SD_FC
    S::AbstractMatrix
    D::AbstractMatrix
end
Flux.@functor SD_FC

(l::SD_FC)(k) = l.S * (l.D' * k)

struct MetaCOMETModel <: AbstractCoMetModel
    # Shared layers
    shared_layers::Vector{NamedTuple{(:S, :V, :D), Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix}}}

    # Paths
    s_dot_path::Flux.Chain
    c_path::Flux.Chain
    nstates::Int
    ncom::Int
end
Flux.@functor MetaCOMETModel


function MetaCOMETModel(nstates::Int, ncom::Int, rank::Int; nhidden::Int=250)
    # Define shared SVD matrices
    shared_layers = [
        (S=randn(Float32, nhidden, rank), V=randn(Float32, rank, rank), D=randn(Float32, nhidden, rank)),
        (S=randn(Float32, nhidden, rank), V=randn(Float32, rank, rank), D=randn(Float32, nhidden, rank))
    ]


    # s_dot_0 path
    s_dot_path = Chain(
        Dense(nstates, nhidden, relu),
        SVD_FC(shared_layers[1].S, shared_layers[1].V, shared_layers[1].D),
        SVD_FC(shared_layers[2].S, shared_layers[2].V, shared_layers[2].D),
        Dense(nhidden, nstates)
    )

    # c path
    c_path = Chain(
        Dense(nstates, nhidden, relu),
        SD_FC(shared_layers[1].S, shared_layers[1].D),
        SD_FC(shared_layers[2].S, shared_layers[2].D),
        Dense(nhidden, ncom)
    )

    return MetaCOMETModel(shared_layers, s_dot_path, c_path, nstates, ncom)
end

function (m::MetaCOMETModel)(states::AbstractArray)
    s_dot_0 = m.s_dot_path(states)
    c = m.c_path(states)
    if m.ncom > 0
        # QR decomposition for orthogonalization
        ∇c = Zygote.jacobian(m.c_path, states)[1]
        A = hcat(∇c, s_dot_0)
        Q, R = qr(A)
        s_dot = Q[:, end] .* R[end, end]
        return s_dot, s_dot_0, c
    else
        return s_dot_0, s_dot_0, c
    end
end

# --- Unified Model Selection ---
struct UnifiedModel
    model::AbstractCoMetModel
end

Flux.@functor UnifiedModel

function select_model(type::Symbol, nstates::Int, ncom::Int; rank::Int=10, use_gpu::Bool=false)
    local model
    if type == :comet
        model = COMETModel(nstates, ncom)
    elseif type == :metacomet
        model = MetaCOMETModel(nstates, ncom, rank)
    else
        error("Invalid model type. Choose :comet or :metacomet")
    end
    if use_gpu
        return UnifiedModel(model |> gpu)
    end
    return UnifiedModel(model)

end
