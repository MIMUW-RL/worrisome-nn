# Note: the CAP functionality has been implemented to be (mostly) readable rather than fast, with a focus on ensuring that the low-level functions
# for running integrators and generating the maps G and DG (written F and DF in this code) are fairly general. The downside is that there are some type 
# instabilities that are unavoidable without significantly more time investment, and this can make the proofs run slower than they should. 
# This is especially the case for anything that communicates with a data file. The symbolic controllers are especially problematic, since each row of the csv
# file requires the controller (and subsequent components) to be recompiled. It would be far more efficient (and transparent, in terms of the code) to use 
# pyCall to grab fields out of the pytorch structures directly and then to parameterize the controllers on the julia side.

function _F(X::Vector{T} where T<:Real, candidate_X::Vector{T} where T<:Real, vector_field, stepper, controller, action_map, η, rotation_function, dimension_space::Int; rotation_amount::T where T<:Real= 2*pi)
    N = length(X);  h = X[1];  x = view(X,2:N);    m = Int64((N-1)/dimension_space);    candidate_x = view(candidate_X,2:N)
    if m*dimension_space+1 !== N
        error("Dimension mismatch: dimension_space = $dimension_space does not divide (length(X)-1).")
    end
    Y = Array{eltype(X)}(undef,N);
    period_offset = rotation_function(rotation_amount)
    state = x[end-dimension_space+1:end]
    action = action_map(controller(state))
    derivative_state = vector_field(state,action)
    stepped_state = stepper(state,derivative_state,h)
    Y[1] = η(x[1:dimension_space],candidate_x[1:dimension_space]); Y[2:1+dimension_space] = x[1:dimension_space] - stepped_state + period_offset
    for n in 2:m
        state = x[dimension_space*(n-2)+1:dimension_space*(n-1)]
        next_state = x[dimension_space*(n-1)+1:dimension_space*(n)]
        action = action_map(controller(state))
        derivative_state = vector_field(state,action)
        stepped_state = stepper(state,derivative_state,h)
        Y[dimension_space*(n-1)+2:dimension_space*n+1] = next_state - stepped_state
    end
    return Y
end

function _F_fix_h(X::Vector{T} where T<:Real,h::T where T<:Real,  vector_field, stepper, controller, action_map, rotation_function, dimension_space::Int; rotation_amount::T where T<:Real= 2*pi)
    m = Int(size(X,1)/dimension_space); N = length(X); 
    if m*dimension_space !== N
        error("Dimension mismatch: dimension space = $dimension_space does not not divide length(X)")
    end
    Y = Array{eltype(X)}(undef,N)
    F = copy(X)
    period_offset = rotation_function(rotation_amount)
    state = X[end-dimension_space+1:end]
    action = action_map(controller(state))
    derivative_state = vector_field(state,action)
    stepped_state = stepper(state,derivative_state,h)
    F[1:dimension_space] = F[1:dimension_space] - stepped_state + period_offset
    for n=1:m
        state = X[dimension_space*(n-1)+1:dimension_space*(n)]
        action = action_map(controller(state))
        derivative_state = vector_field(state,action)
        Y[1 + (n-1)*dimension_space : n*dimension_space] = stepper(state,derivative_state,h)
        if n>1
            F[1 + (n-1)*dimension_space : n*dimension_space] = F[1 + (n-1)*dimension_space : n*dimension_space] - Y[1 + (n-2)*dimension_space : dimension_space*(n-1)]
        end
    end
    return F
end

function _proof_orbit(X,vector_field,stepper,controller,action_map,η,rotation_function,dimension_space,r_star,rotation_amount;verbose=true)
    iX = interval.(X)
    F(x) = _F(x,X,vector_field,stepper,controller,action_map,η,rotation_function,dimension_space;rotation_amount)
    jconfig = ForwardDiff.JacobianConfig(F, iX, ForwardDiff.Chunk{34}());
    Jacobian(x) = ForwardDiff.jacobian( F, x, jconfig)
    ie,Y,Z,msg = check_contraction(iX,F,Jacobian,r_star;Name="Orbit")
    if verbose
        println(msg)
    end
    return ie,Y,Z
end

function _proof_orbit_fix_h(X,h,vector_field,stepper, controller,action_map,rotation_function,dimension_space,r_star,rotation_amount;verbose=true)
    iX = interval.(X);
    F(x) = _F_fix_h(x,h,vector_field,stepper,controller,action_map,rotation_function,dimension_space;rotation_amount)
    jconfig = ForwardDiff.JacobianConfig(F, iX, ForwardDiff.Chunk{34}());
    Jacobian(x) = ForwardDiff.jacobian( F, x, jconfig)
    ie,Y,Z,msg = check_contraction(iX,F,Jacobian,r_star;Name="Orbit")
    if verbose
        println(msg)
    end
    return ie,Y,Z
end

function check_contraction(iX::Vector{Interval{T}} where T<:Real,F::Function,Jacobian::Function,r_star::T where T<:Real;Name="Proof output"::String)
    # Evaluate DF, A.
    DF = wrap(Jacobian(iX));  A = interval.(inv(mid.(DF)));
    # Y bound.
    Y = interval(sup(norm(A*wrap(F(iX)),Inf))) :: eltype(iX)
    # Z bound; inversion error.
    Z₀ = opnorm(I - A*DF,Inf) :: eltype(iX)
    # Z bound; dominant term.
    ball_iX = iX + r_star*interval(-1.0,1.0).*ones(size(iX)[1])
    Z₁₂ = opnorm(A*(wrap(Jacobian(ball_iX)) - DF),Inf) :: eltype(iX)
    # Aggregate Z bound
    Z = interval(sup(Z₀ + Z₁₂));   sZ = sup(Z)
    X = round.(mid.(iX);sigdigits=5)
    if sup(Z)≥1
        msg = Name*" $X: there is no contraction, Z = $sZ."
        return ∅, Y, Z, msg
    else
        r_int = Y / (1-Z);
        r = nextfloat(inf(r_int))
        msg = Name*" $X: validated for all radii r∈[$r,$r_star]."
        return interval(r,r_star), Y, Z, msg
    end
end

function prove_transient(x0,step_size,ep_len,vector_field,stepper,controller,action_map,target_set::Union{Interval,IntervalBox},window_function!,reward_function;final_time="implied",precision::Int=1024)
    if final_time == "implied"
        t_final = step_size*ep_len
    else
        t_final = final_time
    end
    solution = integrate_rigorous(x0,step_size,t_final,vector_field,stepper,controller,action_map;precision=precision)
    solution_window = copy(solution);   
    window_function!(solution_window)
    ind_entry = findfirst(j-> !isempty(IntervalBox(solution_window[:,j]) ∩ IntervalBox(target_set)), 1:size(solution_window,2))
    if isnothing(ind_entry)
        preentry_times = t_final
    else
        preentry_times = (ind_entry-1)*step_size
    end
    total_reward = zero(eltype(solution[1,1]))
    for n in axes(solution,2)
        u = action_map(controller(solution[:,n]))
        total_reward += reward_function(solution[:,n],u)
    end
    return total_reward, preentry_times
end

function prove_transient(x0,step_size,ep_len,vector_field,stepper,controller,action_map,target_set::Tuple{IntervalBox,IntervalBox},window_function!,reward_function;final_time="implied",precision::Int=1024)
    if final_time == "implied"
        t_final = step_size*ep_len
    else
        t_final = final_time
    end
    solution = integrate_rigorous(x0,step_size,t_final,vector_field,stepper,controller,action_map;precision=precision)
    solution_window = copy(solution);   
    window_function!(solution_window)
    t_sets = length(target_set)
    ind_entry = Int64[]
    for n=1:t_sets
        check_containment = findfirst(j-> !isempty(IntervalBox(solution_window[:,j]) ∩ target_set[n]), 1:size(solution_window,2))
        if !isnothing(check_containment)
            push!(ind_entry,check_containment)
        end
    end
    if isempty(ind_entry)
        preentry_times = t_final
    else
        preentry_times = (maximum(ind_entry)-1)*step_size
    end
    total_reward = zero(eltype(solution[1,1]))
    for n in axes(solution,2)
        u = action_map(controller(solution[:,n]))
        total_reward += reward_function(solution[:,n],u)
    end
    return total_reward, preentry_times
end

function wrap(A::AbstractMatrix)
    return LinearOperator(ParameterSpace()^size(A,2),ParameterSpace()^size(A,1),A)
end

function wrap(x::AbstractVector)
    return Sequence(ParameterSpace()^size(x,1),x)
end
