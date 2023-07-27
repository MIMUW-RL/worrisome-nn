function controller₁(X::Vector{T} where T<:Real; control_upper_clip=2.0::T where T<:Real ,control_lower_clip=(-2.0)::T where T<:Real)
    s₁ = cos(X[1]);    s₂ = sin(X[1]);    s₃ = X[2]
    if 0∈s₁
        error("Controller is undefined: 0∈s₁ = $s₁")
    end
    u¹ = -2*s₂ - (8*s₂+2*s₃)/s₁
    if u¹<control_lower_clip
        control_u¹ = control_lower_clip
    elseif u¹ > control_upper_clip
        control_u¹ = control_upper_clip
    elseif control_lower_clip < u¹ < control_upper_clip
        control_u¹ = u¹
    else
        error("Unclipped control u¹ = $u¹ contains a clipping boundary.")
    end
    return control_u¹
end

function controller₂(X::Vector{T} where T<:Real)
    s₁ = cos(X[1]);    s₂ = sin(X[1]);    s₃ = X[2]
    if 0∈s₁
        error("Controller is undefined: 0∈s₁ = $s₁")
    end
    u² = -7.08*s₂ - (13.39*s₂ + 3.12*s₃)/s₁ + 0.27
    control_u² = u²
    return control_u²
end

function controller₂(X::Vector{Interval{T}} where T<:Real)
    s₁ = cos(X[1]);    s₂ = sin(X[1]);    s₃ = X[2]
    if 0∈s₁
        error("Controller is undefined: 0∈s₁ = $s₁")
    end
    u² = -@interval(7.08)*s₂ - (@interval(13.39)*s₂ + @interval(3.12)*s₃)/s₁ + @interval(0.27)
    control_u² = u²
    return control_u²
end

function controller_pendulum_generic(u::Function)
    function Cu(X::Vector{T} where T)
        s₁ = cos(X[1]);    s₂ = sin(X[1]);    s₃ = X[2]
        ret_u = u([s₁;s₂;s₃])
        return ret_u
    end
    return Cu
end

function controller_cartpole_generic(u::Function)
    function Cu(X::Vector{T} where T)
        x = X[1];   x_dot = X[2];   θ = X[3];   θ_dot = X[4]
        control_input = [x;x_dot;cos(θ);sin(θ);θ_dot]
        ret_u = u(control_input)
        return ret_u
    end
    return Cu
end