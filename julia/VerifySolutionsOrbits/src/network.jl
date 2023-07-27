function load_tensors(name::String,folder_path::String,n_layers::Int64)
    W = Vector{Matrix{Float64}}(undef,n_layers)
    B = Vector{Vector{Float64}}(undef,n_layers)
    for k=1:n_layers
        W[k] = readdlm(folder_path*"\\"*name*"."*string(2*(k-1))*".weight.csv", ',', Float64)
        B[k] = vec(readdlm(folder_path*"\\"*name*"."*string(2*(k-1))*".bias.csv", ',', Float64))
    end
    return W, B
end

function load_tensors_zero_bias(name::String,folder_path::String,n_layers::Int64)
    W = Vector{Matrix{Float64}}(undef,n_layers)
    B = Vector{Vector{Float64}}(undef,n_layers)
    for k=1:n_layers
        W[k] = readdlm(folder_path*"\\"*name*string(k)*".weight.csv", ',', Float64)
        B[k] = zeros(size(W[k],1))
    end
    return W, B
end

function convert_weight_bias_bigfloat(W,B;precision=4096)
    setprecision(BigFloat,precision)
    W_big = map(j->big.(W[j]),1:length(W))
    B_big = map(j->big.(B[j]),1:length(B))
    return W_big,B_big
end

function ReLU(x::Real)  
    if x≥0
        return x
    elseif x<0
        return zero(eltype(x))
    else
        error("ReLU argument $x contains zero.")
    end
end

function Sigmoid(x)
    return 1/(1+exp(-x))
end

function scaling_function(x,low::Real,high::Real)
    return low + (x+one(eltype(low)))*(high - low)/(2*one(eltype(low)))
end

function Network(x, Weight_tensors, Bias_tensors, activation::Vector{String}, scaling::Function)
    n_layers = size(Bias_tensors)[1]
    if n_layers != size(activation)[1]
        s_activation = size(activation)[1]
        error("Activation string (length $s_activation) is incompatible with number of layers ($n_layers layers).")
    end
    for k=1:n_layers
        x = Weight_tensors[k]*x + Bias_tensors[k]
        if activation[k]=="ReLU"
            x = ReLU.(x)
        elseif activation[k]=="Sigmoid"
            x = Sigmoid.(x)
        elseif activation[k]=="Tanh"
            x = tanh.(x)
        end
    end
    return scaling(x[1])
end