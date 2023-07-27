# Structs

Base.@kwdef struct PendulumParams{T<:Real}
    g::T=10.0
    ℓ::T=1.0
    m::T=1.0
end

function Interval_PendulumParams()
    g = @interval(10.0)
    ℓ = @interval(1.0)
    m = @interval(1.0)
    return PendulumParams(g,ℓ,m)
end

function Base.show(io::IO, ::MIME"text/plain", P::PendulumParams{T}) where {T}
    g = P.g;  ℓ = P.ℓ;    m = P.m
    println(io, "PendulumParams{$T} with fields:") 
    println(io,"g = $g,")
    println(io,"ℓ = $ℓ,")
    print(io,"m = $m.")
end

# Integrator function constructor.

function pendulum_steppers(;velocity_lower_clip=-8.0, velocity_upper_clip=8.0, left_boundary_action=-2.0, right_boundary_action=+2.0 )
    if !(left_boundary_action < right_boundary_action)
        error("Keyword arguments 'left_boundary' and 'right_boundary' are not ordered correctly.")
    end
    if !(velocity_lower_clip < velocity_upper_clip) 
        error("Keyword arguments 'velocity_lower_clip' and 'velocity_upper_clip' are not ordered correctly.")
    end
    function explicit_step(state,derivative_state,step_size)
        update = state + derivative_state*step_size
        if update[2] < velocity_lower_clip
            update[2] = velocity_lower_clip*one(typeof(update[2]))
        elseif update[2] > velocity_upper_clip
            update[2] = velocity_upper_clip*one(typeof(update[2]))
        elseif (velocity_lower_clip < update[2]) && (update[2] < velocity_upper_clip)
            # do nothing.
        else
            error("Unclipped velocity intersects boundary {$velocity_lower_clip,$velocity_upper_clip}.")
        end
        return update
    end
    function semi_implicit_step(state,derivative_state,step_size)
        theta,theta_dot = state
        _,thetadot_update = derivative_state
        ωₖ_unclipped = theta_dot + step_size*thetadot_update
        if ωₖ_unclipped < velocity_lower_clip
            ωₖ = velocity_lower_clip
        elseif ωₖ_unclipped > velocity_upper_clip
            ωₖ = velocity_upper_clip
        elseif velocity_lower_clip < ωₖ_unclipped < velocity_upper_clip
            ωₖ = ωₖ_unclipped
        else
            error("Unclipped velocity boundary {$velocity_lower_clip,$velocity_upper_clip}.")
        end
        θₖ = theta + step_size*ωₖ
        return [θₖ;ωₖ]
    end
    function action_logic(action)
        if action<left_boundary_action
            return one(typeof(action))*left_boundary_action
        elseif action > right_boundary_action
            return one(typeof(action))*right_boundary_action
        elseif (left_boundary_action < action) && (action < right_boundary_action)
            return action
        else
            error("Action intersects boundary {$left_boundary_action,$right_boundary_action}.")
        end
    end
    return explicit_step,semi_implicit_step,action_logic
end

# Vector fields.

function _pendulum_vector_field(x::Vector{T} where T, action::T where T; params::PendulumParams=PendulumParams())
    return [x[2] ; 3*action/(params.m*params.ℓ^2) + 3*params.g*sin(x[1])/(2*params.ℓ)]
end

function pendulum_vector_field( ;params::PendulumParams=PendulumParams())
    return (x,u) -> _pendulum_vector_field(x,u; params=params)
end

# Reward/penalty/proof functions

function solution_window_pendulum!(x)
    x[1,:] = acos.(cos.(x[1,:]))
end

function pendulum_reward_function(X,u;weight₁ = @interval(0.1), weight₂ = @interval(0.001))
    x,dθ = X
    x_normalized = acos(cos(x))
    reward = -(x_normalized^2 + weight₁*(dθ^2) + weight₂*(u^2))
    return reward
end

function rotation_function_pendulum(x)
    return [x;0]
end

function η_pendulum(x,X)    # Fixes the "θ" coordinate.
    return x[1]-X[1]
end