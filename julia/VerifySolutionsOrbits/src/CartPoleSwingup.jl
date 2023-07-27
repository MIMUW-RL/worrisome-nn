# Structs.

Base.@kwdef struct CartParams{T<:Real}
    width::T = 1/3
    height::T = 1/6
    mass::T = 0.5
end

Base.@kwdef struct PoleParams{T<:Real}
    width::T = 0.05
    length::T = 0.6
    mass::T = 0.5
end

Base.@kwdef struct CartPoleSwingupParams{T<:Real}
    gravity::T = 9.82
    forcemag::T = 10.0
    friction::T = 0.1
    cart::CartParams{T} = CartParams()
    pole::PoleParams{T} = PoleParams()
    masstotal::T
    mpl::T
end

function CartPoleSwingupParams(;gravity=9.82,forcemag=10.0,friction=0.1,cart::CartParams=CartParams(),pole::PoleParams=PoleParams())
    masstotal = cart.mass + pole.mass
    mpl = pole.mass*pole.length
    return CartPoleSwingupParams(gravity,forcemag,friction,cart,pole,masstotal,mpl)
end

function Base.show(io::IO, ::MIME"text/plain", P::CartPoleSwingupParams{T}) where {T}
    println(io, "CartPoleSwingupParams{$T} with fields:")
    gravity = P.gravity;  forcemag = P.forcemag;    friction = P.friction;  masstotal = P.masstotal;  mpl = P.mpl;
    println(io, "gravity = $gravity",)
    println(io, "forcemag = $forcemag,")
    println(io, "friction = $friction,")
    println(io, "masstotal = $masstotal,")
    print(io,"masspole×length (mpl) = $mpl.")
end

function Interval_CartPoleSwingupParams(;precision=512)
    setprecision(BigFloat,precision)
    gravity = @biginterval(9.82)
    forcemag = @biginterval(10)
    friction = @biginterval(1/10)
    cart_width = @biginterval(1/3)
    cart_height = @biginterval(1/6)
    cart_mass = @biginterval(1/2)
    cart = CartParams(cart_width,cart_height,cart_mass)
    pole_width = @biginterval(0.05)
    pole_length = @biginterval(0.6)
    pole_mass = @biginterval(0.5)
    pole = PoleParams(pole_width,pole_length,pole_mass)
    return CartPoleSwingupParams(;gravity=gravity,forcemag=forcemag,friction=friction,cart=cart,pole=pole)
end

# Integrator function constructor / other integrator functions.

function CartPoleSwingup_steppers( ; left_boundary_action=-1,right_boundary_action=+1)
    if !(left_boundary_action < right_boundary_action)
        error("Keyword arguments 'left_boundary_action' and 'right_boundary_action' are not ordered correctly.")
    end
    function explicit_step(state,derivative_state,step_size)
        return state + derivative_state*step_size
    end
    function semi_implicit_step(state,derivative_state,step_size)
        x_pos,x_dot,theta,theta_dot = state
        _,xdot_update,_,thetadot_update = derivative_state
        return [ 
            x_pos + (x_dot + xdot_update*step_size)*step_size ; 
            x_dot + xdot_update*step_size ; 
            theta + (theta_dot + thetadot_update*step_size)*step_size
            theta_dot + thetadot_update*step_size
            ]
    end
    function action_logic(action)
        if action<left_boundary_action
            return one(typeof(action))*left_boundary_action
        elseif action > right_boundary_action
            return one(typeof(action))*right_boundary_action
        elseif (left_boundary_action < action) && (action < right_boundary_action)
            return action
        else
            error("Action is ambiguous; intersects boundary actions {$left_boundary_action,$right_boundary_action}.")
        end
    end
    return explicit_step,semi_implicit_step,action_logic
end

function action_map_cartpole_NN(x)  # Identity.
    return x
end

# Vector Field.

function _cart_pole_swingup_vector_field(X::Vector{T} where T, action::T where T ; params::CartPoleSwingupParams=CartPoleSwingupParams())
    _,x_dot,theta,theta_dot = X
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    action = action*params.forcemag

    xdot_update = (
        -2 * params.mpl * (theta_dot^2) * sin_theta
        + 3 * params.pole.mass * params.gravity * sin_theta * cos_theta
        + 4 * action
        - 4 * params.friction * x_dot
    ) / (4 * params.masstotal - 3 * params.pole.mass * cos_theta^2)

    thetadot_update = (
        -3 * params.mpl * (theta_dot^2) * sin_theta * cos_theta
        + 6 * params.masstotal * params.gravity * sin_theta
        + 6 * (action - params.friction * x_dot) * cos_theta
    ) / (4 * params.pole.length * params.masstotal - 3 * params.mpl * cos_theta^2)
    return [x_dot;xdot_update;theta_dot;thetadot_update]
end

function cart_pole_swingup_vector_field(;params::CartPoleSwingupParams=CartPoleSwingupParams())
    return (x,u) -> _cart_pole_swingup_vector_field(x,u;params)
end

# Penalty/proof function

function cartpole_penalty_function(X,u;weight₁ = @interval(0.5), weight₂ = @interval(0.5))
    x,x_dot,theta,theta_dot = X
    penalty = -cos(theta) + weight₁*theta_dot^2 + weight₂*x_dot^2
    return penalty
end

function rotation_function_cartpole(x)
    return [0;0;x;0]
end

function η_cartpole(x,X)    # Fixes the "x" coordinate.
    return x[1]-X[1]
end