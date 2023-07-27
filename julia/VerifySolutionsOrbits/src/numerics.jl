function step_interval_lookup(x)
    if 0.049<x && x<0.051
        return @interval("0.05")
    elseif 0.0249<x && x<0.0251
        return @interval("0.025")
    elseif 0.01249<x && x<0.01251
        return @interval("0.0125")
    elseif 0.009<x && x<0.011
        return @interval("0.01")
    elseif 0.0009<x && x<0.0011
        return @interval("0.001")
    elseif 0.0049<x && x<0.0051
        return @interval("0.005")
    elseif 0.00249<x && x<0.00251
        return @interval("0.0025")
    else
        error("Input was not sufficiently close to one of 0.01, 0.001, 0.05, 0.005, 0.025, 0.0025 or 0.0125")
    end
end

# Search functions for numerical orbits.

function find_orbit(X;N=1,direction=+1,verbose=false)
    warn=false
    if direction==+1
        if !isa(findlast(X[1,:].≤X[1,end]-2π*N),Nothing)
            t_max_orbit = findlast(X[1,:].≤X[1,end]-N*2π)
            orbit = X[:,t_max_orbit+1:end]
            shift = floor(orbit[1,end]/(2π))
            orbit[1,:] = orbit[1,:] .- (shift-1)*2π
        else
            if verbose
                @warn "find_orbit() failed to find an orbit!"
            end
            warn=true
            orbit = X
        end
    elseif direction==-1
        if !isa(findlast(X[1,:].≥X[1,end]+2π*N),Nothing)
            t_max_orbit = findlast(X[1,:].≥X[1,end]+N*2π)
            orbit = X[:,t_max_orbit+1:end]
            shift = floor(orbit[1,end]/(2π))
            orbit[1,:] = orbit[1,:] .- (shift-1)*2π
        else
            if verbose
                @warn "find_orbit() failed to find an orbit!"
            end
            warn=true
            orbit = X
        end
    else error("Bad direction.")
    end
    return orbit,warn
end

function find_orbit_neutral(X,radius)
    dim = size(X[:,1])
    ball = ones(dim)*interval(-radius,radius)
    terminal_point = X[:,end]
    target = terminal_point + ball
    indices = findall( map(j->all(X[:,j] .∈ target) , axes(X,2)) )
    last_jump_temp = findlast(diff(indices) .!= 1)
    last_jump = indices[last_jump_temp]
return X[:,last_jump:end]
end

# Convenient generation of (F,DF) pairs.

function get_F_DF(candidate_X,vector_field, stepper, controller, action_map, η, rotation_function, dimension_space::Int; rotation_amount::T where T<:Real= 2*pi)
    F(x) = _F(x,candidate_X,vector_field, stepper, controller, action_map, η, rotation_function, dimension_space; rotation_amount=rotation_amount)
    DF(x) = ForwardDiff.jacobian( F, x )
    return x -> (F(x), DF(x))
end

function get_F_DF_fix_h(h::T where T<:Real,  vector_field, stepper, controller, action_map, rotation_function, dimension_space::Int; rotation_amount::T where T<:Real= 2*pi)
    F(x) = _F_fix_h(x, h, vector_field, stepper, controller, action_map, rotation_function, dimension_space; rotation_amount=rotation_amount)
    DF(x) = ForwardDiff.jacobian( F, x )
    return x -> (F(x), DF(x))
end

# Generic Newton's method

function Newton(X::Vector{T} where T<:Real,F_DF::Function;tol::Float64=5E-13,max_iter::Int64=32,verbose=true)
    f,df = F_DF(X);   idf = inv(df);  ΔX = -idf*f; defect = norm(ΔX,Inf); F_defect = norm(f,Inf)
    if verbose
        println("⋅ Newton iteration 0: |AF| = $defect, |F| = $F_defect")
    end
    X0 = copy(X)
    iter = 1
    while defect > tol && iter < max_iter+1 && defect < 1E5
        X = X + ΔX
        f,df = F_DF(X);       
        idf = inv(df);  ΔX = -idf*f; defect = norm(ΔX,Inf);   F_defect = norm(f,Inf)
        if verbose==true
            println("⋅ Newton iteration $iter: |AF| = $defect, |F| = $F_defect")
        end
        iter += 1
    end
    correction_amount = norm(X0-X,Inf)
    if verbose
        println("⋅ Newton final defect: |AF| = $defect, |F| = $F_defect, norm total correction = $correction_amount.")
    end
    return X,defect,correction_amount
end

function penalty_escape(x0,step_size,ep_len,vector_field,stepper,controller,action_map,target_set::Union{Interval,IntervalBox},window_function!,reward_function;final_time="implied")
    if final_time == "implied"
        t_final = step_size*ep_len
    else
        t_final = final_time
    end
    solution = integrate(x0,step_size,t_final,vector_field,stepper,controller,action_map)
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

function penalty_escape(x0,step_size,ep_len,vector_field,stepper,controller,action_map,target_set::Tuple{IntervalBox,IntervalBox},window_function!,reward_function;final_time="implied")
    if final_time == "implied"
        t_final = step_size*ep_len
    else
        t_final = final_time
    end
    solution = integrate(x0,step_size,t_final,vector_field,stepper,controller,action_map)
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

function unique_orbit(list_orbit,index)
    orbit_indices = map(j->isassigned(list_orbit,j),1:length(list_orbit))
    orbit = list_orbit[index]
    index = sum(orbit_indices[1:index])
    list_orbit = list_orbit[orbit_indices]
    if index==1
        list_orbit = list_orbit[2:end]
    elseif index==length(list_orbit)
        list_orbit = list_orbit[1:end-1]
    else
        list_orbit = [list_orbit[1:index-1];list_orbit[index+1:end]]
    end
    N = length(list_orbit)
    unique = true
    duplicates = Int64[]
    for k=1:N
        if length(orbit)==length(list_orbit[k])
            if iseven(length(orbit))
                diffθ = mod.(orbit[1],2π) .- mod.(list_orbit[k][1:2:end],2π)
                diffω = orbit[2] .- list_orbit[k][2:2:end]
            elseif isodd(length(orbit))
                diffθ = mod.(orbit[2],2π) .- mod.(list_orbit[k][2:2:end],2π)
                diffω = orbit[3] .- list_orbit[k][3:2:end]
            end
            diff = [diffθ diffω]
            norm_diff = map(j->norm(diff[j,:]),1:size(diff,1))
            if minimum(norm_diff)<1e-4
                unique = false
                if index==1
                    push!(duplicates,k+1)
                elseif index==length(list_orbit)
                    push!(duplicates,k)
                elseif k<index
                    push!(duplicates,k)
                else
                    push!(duplicates,k+1)
                end
            end
        end
    end
    if isempty(duplicates)
        duplicates = [index]
    else
        duplicates = sort(push!(duplicates,index))
    end
    return unique, duplicates
end