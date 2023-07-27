# Integrator.

function integrate(x_ic::Vector{T} where T<:Real, step_size::Real, t_final::Real, vector_field, stepper, controller, action_map)
    N = Int(ceil(t_final/step_size))
    X = Array{eltype(x_ic)}(undef,size(x_ic,1),N+1);  X[:,1] = x_ic
    for n in 2:N+1
        action = action_map(controller(X[:,n-1]))
        Xdot = vector_field(X[:,n-1],action)
        X[:,n] = stepper(X[:,n-1],Xdot,step_size)
    end
    return X
end

function integrate_rigorous(x_ic::Vector{T} where T, step_size::Real, t_final::Real, vector_field, stepper, controller, action_map;precision::Int=512)
    setprecision(BigFloat,precision)
    x_ic_int = x_ic .+ @biginterval(0)
    X = integrate(x_ic_int,step_size,t_final,vector_field,stepper,controller,action_map)
    return X
end