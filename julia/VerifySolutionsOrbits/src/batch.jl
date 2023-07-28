# Batch proofs and calculations.

function prove_transients_pendulum(csv_string::String; stabilized_set = IntervalBox(interval(-1e-2,1e-2),interval(-1e-2,1e-2)), 
    final_time = "implied", precision=2048, proof=true, vector_field=pendulum_vector_field(params=PendulumParams()), 
    explicit_implicit_actionmap = pendulum_steppers(), η = η_pendulum)
    ## DESCRIPTION ##
    # Takes as input a formatted csv file and checks each row (model/unstabilized solution) for existence of a periodic orbit, using    
    # our heuristic search. If found, proves the orbit and stores a proof certificate. If no is orbit found, we consider the solution
    # to be unstabilized, and rigorously compute reward.
    # Inputs:
    #   csv_string::String -- path to a csv file
    # Optional:
    #   stabilized_set::IntervalBox -- "stabilized" interval IntervalBox
    #   final_time::Union{Real,String} -- if "implied", treats final time as being implied by csv file, which includes an episode length
    #       field and step size. If a real-type number, treats that number as the final time.
    #   others -- should not be changed unless you know what you're doing.
    # Outputs:
    #   tuple: solutions, orbits, stabilized_flag(bool), exist_orbit(bool), returns, Tp, max_reward,
    #           proofs, defects, corrections.
    explicit_stepper, implicit_stepper, action_map = explicit_implicit_actionmap
    dataset = DataFrame(CSV.File(csv_string))
    solutions = Vector{Matrix{Float64}}(undef,size(dataset,1))
    orbits = Vector{Vector{Float64}}(undef,size(dataset,1))
    returns = Vector{Interval}(undef,size(dataset,1))
    max_reward = Vector{Interval}(undef,size(dataset,1))
    Tp = Vector{Float64}(undef,size(dataset,1))
    flag = Vector{String}(undef,size(dataset,1))
    defects = Vector{Float64}(undef,size(dataset,1))
    corrections = Vector{Float64}(undef,size(dataset,1))
    proofs = Vector{Interval}(undef,size(dataset,1))
    exist_orbit = Vector{Bool}(undef,size(dataset,1))
    for k=1:size(solutions,1)
        x0 = [dataset.init_omega[k];dataset.init_theta[k]]
        symbolic_expression = "(x0,x1,x2)->"*dataset.formula[k]
        func = eval(Meta.parse(symbolic_expression))
        parsed_controller(x) = Base.invokelatest(func,x[1],x[2],x[3])
        if !("explicit" in names(dataset))
            method = implicit_stepper
        elseif dataset.explicit[k] == true
            method = explicit_stepper
        else
            method = implicit_stepper
        end
        step_size = dataset.h[k]
        if final_time == "implied"
            t_final = step_size*dataset.ep_len[k]
        else
            t_final = final_time
        end
        solutions[k]= integrate(x0,step_size,t_final,vector_field,method,controller_pendulum_generic(parsed_controller),action_map)
        println("Transient $k: rigorously computing return.")
        returns[k],Tp[k] = prove_transient(x0,step_size,dataset.ep_len[k],vector_field,method,controller_pendulum_generic(parsed_controller),action_map,stabilized_set,solution_window_pendulum!,pendulum_reward_function;final_time,precision)
        ret = Int(round(mid(returns[k])))
        println("Transient $k: return = $ret computed. Starting orbit search.")
        if solutions[k][1,end]<0
            direction = -1
            d = "Clockwise"
            rotation = -2π
            irotation = -@interval(2π)
        else 
            direction = +1
            d = "Counter-Clockwise"
            rotation = +2π
            irotation = @interval(2π)
        end
        if norm(solutions[k][2,end-10:end])>1e-1
            Y,warn = find_orbit(solutions[k];direction=direction)
            Y = reshape(Y,2*size(Y,2))
            if !warn
                F_DF = get_F_DF_fix_h(step_size,  vector_field, method, controller_pendulum_generic(parsed_controller), action_map, rotation_function_pendulum, 2; rotation_amount=rotation)
                OUT = try Newton(Y,F_DF;verbose=false)
                catch 
                    ([Inf],1.0,Inf)
                end
                orbits[k] = OUT[1]; defect = OUT[2]; defects[k] = defect;   corrections[k] = OUT[3];
                if defect<1E-10
                    cor = OUT[3];
                    flag[k] = "Fixed_"*d
                    println("Transient $k: $d orbit found at fixed step size $step_size, Newton correction (norm) = $cor.")
                    exist_orbit[k] = true
                    if proof
                        ie = ∅
                        r★=1E-4
                        while isempty(ie) && r★>1E-12
                            prf = try _proof_orbit_fix_h(orbits[k],step_interval_lookup(step_size),vector_field,method,controller_pendulum_generic(parsed_controller),action_map,rotation_function_pendulum,2,r★,irotation;verbose=false)
                            catch
                                (∅,1.0,1.0)
                            end
                            ie = prf[1]
                            r★ = r★/10
                        end
                        proofs[k]=ie
                        if !isempty(ie)
                            println("Proof successful; existence interval = $ie.")
                        else
                            println("Proof failed for r★≥$r★")
                        end
                    end
                else
                    F_DF = get_F_DF([step_size;Y],vector_field,method,controller_pendulum_generic(parsed_controller),action_map,η,rotation_function_pendulum,2;rotation_amount=rotation)
                    OUT = try Newton([step_size;Y],F_DF;verbose=false)
                    catch
                        ([Inf],1.0,Inf)
                    end
                    orbits[k] = OUT[1]; defect = OUT[2]; defects[k] = defect;   corrections[k] = OUT[3];
                    if defect<1E-10
                        cor = OUT[3];
                        flag[k] = "Unfixed_"*d
                        stepsize_new = orbits[k][1]
                        println("Transient $k: $d orbit found at unfixed step size $step_size, Newton correction (norm) = $cor. Step size corrected to $stepsize_new.")
                        exist_orbit[k] = true
                        if proof
                            ie = ∅
                            r★=1E-4
                            while isempty(ie) && r★>1E-12
                                prf = try 
                                    _proof_orbit(orbits[k],vector_field,method,controller_pendulum_generic(parsed_controller),action_map,η,rotation_function_pendulum,2,r★,irotation;verbose=false)
                                catch
                                    (∅,1.0,1.0)
                                end
                                ie = prf[1]
                                r★ = r★/10
                            end
                            proofs[k]=ie
                            if !isempty(ie)
                                println("Proof successful; existence interval = $ie.")
                            else
                                println("Proof failed for r★≥$r★")
                            end
                        end
                    else
                        flag[k] = "Diverged"
                        println("Transient $k: failed to find a rotating orbit; Newton did not converge.")
                        exist_orbit[k] = false
                    end
                end
            else
                println("Transient $k: norm of solution near end of integration suggests an orbit, but did not find a rotating one.")
                flag[k] = "No Orbit"
                exist_orbit[k] = false
            end
        else
            println("Transient $k: norm of solution near end of integration exceeds tolerance; not likely an orbit.")
            flag[k] = "No Orbit"
            exist_orbit[k] = false
        end
        if exist_orbit[k] && proof
            if iseven(length(orbits[k]))
                offset = 0
            else
                offset = 1
            end
            orbit_enc = orbits[k] .+ Interval(-inf(proofs[k]),inf(proofs[k]))
            all_rewards = map( j->pendulum_reward_function(orbit_enc[offset+2*j-1:offset+2*j], action_map(controller_pendulum_generic(parsed_controller)(orbit_enc[offset+2*j-1:offset+2*j])  )), 1:Int(floor(length(orbit_enc)/2)) )
            max_reward[k] = maximum(all_rewards)
        end
    end
    return solutions, orbits, flag, exist_orbit, returns, Tp, max_reward, proofs, defects, corrections
end

function proof_cartpole_LargeNet(jld2_file::String,tensors_folder::String)
    iP = Interval_CartPoleSwingupParams()
    ivf = cart_pole_swingup_vector_field(params=iP)
    expl,impl,_ = CartPoleSwingup_steppers()
    Orbit = load(jld2_file)["orbit_flat_corrected"]
    method = load(jld2_file)["method"]
    r★ = load(jld2_file)["r★"]
    W,B = load_tensors("td3_actor_400_300_ReLU.pth_mu",tensors_folder,3)
    @inline N(x) = Network(x,W,B,["ReLU";"ReLU";"Tanh"],y->y)
    controller = controller_cartpole_generic(N)
    ie = Vector{Interval{Float64}}(undef,length(Orbit))
    max_abs_θ = Vector{Float64}(undef,length(Orbit))
    ϵ = Vector{Float64}(undef,length(Orbit))
    penalty_accumulation = Vector{Interval{Float64}}(undef,length(Orbit))
    mean_penalty = Vector{Interval{Float64}}(undef,length(Orbit))
    for n in axes(Orbit,1)
        printstyled("Proving orbit $n, with step size h = ",Orbit[n][1]," and integration method = ",method[n],"... \n",color=:blue)
        if method[n]=="explicit"
            ie[n],_,_ = _proof_orbit(Orbit[n],ivf,expl,controller,action_map_cartpole_NN,η_cartpole,rotation_function_cartpole,4,r★[n],0;verbose=false)
        else
            ie[n],_,_ = _proof_orbit(Orbit[n],ivf,impl,controller,action_map_cartpole_NN,η_cartpole,rotation_function_cartpole,4,r★[n],0;verbose=false)
        end
        println("Existence interval: ",ie[n])
        println("Computing penalty data over orbit...")
        error_ball = sup(ie[n])*interval(-1,1)
        S = interval.(reshape(Orbit[n][2:end],4,Int(length(Orbit[n][2:end])/4))) .+ error_ball
        p = interval(0)
        for m in axes(S,2)
            u = controller((S[:,m]))
            p += cartpole_penalty_function(interval.(S[:,m]),u)
        end
        max_abs_θ[n] = maximum(sup.(abs.(S[3,:])))
        penalty_accumulation[n] = p
        mean_penalty[n] = p/size(S,2)
        ϵ[n] = minimum(map(j->inf(norm(S[[2;3;4],j],Inf)) ,1:size(S,2)  ))
        println("Accumulated penalty: ",p)
        println("Mean penalty: ", mean_penalty[n])
        println("Maximum amplitude (θ): ",max_abs_θ[n])
        println("ϵ level: ",ϵ[n])
    end
    return ie,penalty_accumulation,mean_penalty,max_abs_θ,ϵ
end

function prove_transients_cartpole(csv_string::String; ep_len=2000, precision=4096, vector_field=cart_pole_swingup_vector_field(params=CartPoleSwingupParams_Base()),
    interval_vector_field=cart_pole_swingup_vector_field(params=Interval_CartPoleSwingupParams(precision=precision)), explicit_implicit_actionmap = CartPoleSwingup_steppers(),
    xbound_left = -2.4, xbound_right = 2.4)
    ## DESCRIPTION ##
    # Proves unstabilized solutions (previously we called them transients) for the cart-pole model with controller (symbolic) specified in a csv.
    # Inputs:
    #   csv_string::String -- path to a csv file
    # Optional (has defaults):
    #   ep_len::Int64 -- episode length
    #   precision::Int64 -- numerical precision used for interval arithmetic. Has been set very high to ensure all proofs are successful, but can be set lower in some cases.
    #   others -- should not be changed unless you know what you're doing.
    explicit_stepper, implicit_stepper, action_map = explicit_implicit_actionmap
    dataset = DataFrame(CSV.File(csv_string))
    solutions = Vector{Matrix{Float64}}(undef,size(dataset,1))
    escape = Vector{Float64}(undef,size(dataset,1))
    escape_flag = Vector{Bool}(undef,size(dataset,1))
    penalty = Vector{Interval}(undef,size(dataset,1))
    ℛ = interval(-Inf,Inf)
    escape_left_x = IntervalBox(interval(-Inf,xbound_left),ℛ,ℛ,ℛ)
    escape_right_x = IntervalBox(interval(xbound_right,Inf),ℛ,ℛ,ℛ)
    escape_set =  escape_left_x , escape_right_x
    for k=1:size(solutions,1)
        x0 = [dataset.init_xpos[k];dataset.init_xdot[k];dataset.init_theta[k];dataset.init_thetadot[k]]
        symbolic_expression = "(x0,x1,x2,x3,x4)->"*dataset.formula[k]
        func = eval(Meta.parse(symbolic_expression))
        parsed_controller(x) = Base.invokelatest(func,x[1],x[2],x[3],x[4],x[5])
        if !("explicit" in names(dataset))
            method = implicit_stepper
        elseif dataset.explicit[k] == true
            method = explicit_stepper
        else
            method = implicit_stepper
        end
        step_size = dataset.h[k]
        t_final = step_size*ep_len
        solutions[k]= integrate(x0,step_size,t_final,vector_field,method,controller_cartpole_generic(parsed_controller),action_map)
        println("Transient $k: rigorously computing penalty.")
        penalty[k],escape[k] = prove_transient(x0,step_size,ep_len,interval_vector_field,method,controller_cartpole_generic(parsed_controller),action_map,escape_set,x->x,cartpole_penalty_function;precision,final_time=t_final)
        if escape[k]<t_final
            escape_flag[k] = true
        else
            escape_flag[k] = false
        end
    end
    return solutions, penalty, escape, escape_flag
end

function calculate_transients_cartpole_NeuralNet(controller, csv_string::String; ep_len=2000, vector_field=cart_pole_swingup_vector_field(params=CartPoleSwingupParams_Base()),
    explicit_implicit_actionmap = CartPoleSwingup_steppers())
    # Calculates penalties (non-rigorous) associated to a neural network controller and csv file of unstabilized solutions.
    # This function is not used in the paper, and is included for reader interest only.
    explicit_stepper, implicit_stepper, _ = explicit_implicit_actionmap
    dataset = DataFrame(CSV.File(csv_string))
    solutions = Vector{Matrix{Float64}}(undef,size(dataset,1))
    escape = Vector{Float64}(undef,size(dataset,1))
    escape_flag = Vector{Bool}(undef,size(dataset,1))
    penalty = Vector{Interval}(undef,size(dataset,1))
    ℛ = interval(-Inf,Inf)
    escape_left_x = IntervalBox(interval(-Inf,-2.4),ℛ,ℛ,ℛ)
    escape_right_x = IntervalBox(interval(2.4,Inf),ℛ,ℛ,ℛ)
    escape_set =  escape_left_x , escape_right_x
    for k=1:size(solutions,1)
        x0 = [dataset.init_xpos[k];dataset.init_xdot[k];dataset.init_theta[k];dataset.init_thetadot[k]]
        if !("explicit" in names(dataset))
            method = implicit_stepper
        elseif dataset.explicit[k] == true
            method = explicit_stepper
        else
            method = implicit_stepper
        end
        step_size = dataset.h[k]
        t_final = step_size*ep_len
        solutions[k]= integrate(x0,step_size,t_final,vector_field,method,controller_cartpole_generic(controller),action_map_cartpole_NN)
        penalty[k],escape[k] = penalty_escape(x0,step_size,ep_len,vector_field,method,controller_cartpole_generic(controller),action_map_cartpole_NN,escape_set,x->x,cartpole_penalty_function;final_time=t_final)
        if escape[k]<t_final
            escape_flag[k] = true
        else
            escape_flag[k] = false
        end
    end
    return solutions, penalty, escape_flag, escape, dataset.h
end

function prove_transients_cartpole_NeuralNet(controller, csv_string::String; ep_len=2000, vector_field=cart_pole_swingup_vector_field(params=CartPoleSwingupParams_Base()),precision=4096,
    interval_vector_field=cart_pole_swingup_vector_field(params=Interval_CartPoleSwingupParams(precision=precision)), explicit_implicit_actionmap = CartPoleSwingup_steppers())
    # Calculates penalties and data rigorously for unstabilized solutions for specified neural net controller for cartpole swingup.
    # Input:
    # controller::Function -- Should behave like a map from ℛ⁵→ℛ. Usually this is something like x->Network(x,W,B,activation,scaling) for suitable activation and scaling functions.
    # csv_string::String -- path to a csv file, with expected formatting.
    # Output (tuple): solutions, accumulated penalties, escape time, escape flag(bool), step size.
    ## DESCRIPTION ##
    explicit_stepper, implicit_stepper, _ = explicit_implicit_actionmap
    dataset = DataFrame(CSV.File(csv_string))
    solutions = Vector{Matrix{Float64}}(undef,size(dataset,1))
    escape = Vector{Float64}(undef,size(dataset,1))
    escape_flag = Vector{Bool}(undef,size(dataset,1))
    penalty = Vector{Interval}(undef,size(dataset,1))
    ℛ = interval(-Inf,Inf)
    escape_left_x = IntervalBox(interval(-Inf,-2.4),ℛ,ℛ,ℛ)
    escape_right_x = IntervalBox(interval(2.4,Inf),ℛ,ℛ,ℛ)
    escape_set =  escape_left_x , escape_right_x
    for k=1:size(solutions,1)
        x0 = [dataset.init_xpos[k];dataset.init_xdot[k];dataset.init_theta[k];dataset.init_thetadot[k]]
        if !("explicit" in names(dataset))
            method = implicit_stepper
        elseif dataset.explicit[k] == true
            method = explicit_stepper
        else
            method = implicit_stepper
        end
        step_size = dataset.h[k]
        t_final = step_size*ep_len
        solutions[k]= integrate(x0,step_size,t_final,vector_field,method,controller_cartpole_generic(controller),action_map_cartpole_NN)
        println("Transient $k: rigorously computing penalty.")
        penalty[k],escape[k] = prove_transient(x0,step_size,ep_len,interval_vector_field,method,controller_cartpole_generic(controller),action_map_cartpole_NN,escape_set,x->x,cartpole_penalty_function;final_time=t_final,precision=precision)
        if escape[k]<t_final
            escape_flag[k] = true
        else
            escape_flag[k] = false
        end
    end
    return solutions, penalty, escape, escape_flag, dataset.h
end

function Landajuela_proofs_LaTeXString(orbits_path)
    # Outputs the table associated to the Landajuela et. al periodic orbits. Also putputs [r*,max_r] for validation.
    # Relative to the repo, files are contained in \julia\orbits_data_Landajuela_controller.
    # Input: orbits_path::String -- path to the folder containg the Landajuela et. al orbits (jld2 files).
    vector_field=pendulum_vector_field(params=PendulumParams())
    explicit_stepper, implicit_stepper, action_map = pendulum_steppers()
    list = readdir(orbits_path)
    N = length(list)
    str_orbit = ""
    fail = Vector{Bool}(undef,N)
    # Re-order the list for prettyness later.
    list_explicit = String[]
    list_implicit = String[]
    for k=1:N
        if occursin("Explicit",list[k])
            push!(list_explicit,list[k])
        else
            push!(list_implicit,list[k])
        end
    end
    list = [sort(list_explicit;rev=true);sort(list_implicit;rev=true)]
    for k=1:N
        save_file = orbits_path*"\\"*list[k]
        X = load(save_file)["X"];   step_size = load(save_file)["h"]
        m = Int64(length(X)/2)
        θ₀ = round(X[1],digits=5)
        ω₀ = round(X[2],digits=5)
        h = round(step_size,digits=5)
        if occursin("Explicit",list[k])
            method = explicit_stepper
            method_str = "Explicit"
        else
            method = implicit_stepper
            method_str = "Semi-Implicit"
        end
        ie = ∅
        r★=1E-4
        while isempty(ie) && r★>1E-12
            printstyled("Attempting ";color= :yellow )
            println("proof of "*list[k]*" at r★=$r★.")
            prf = try 
                _proof_orbit_fix_h(X,step_interval_lookup(step_size),vector_field, method, controller₂, action_map, rotation_function_pendulum, 2, r★, @interval(2pi); verbose=false)
            catch
                (∅,1.0,1.0)
            end
            ie = prf[1]
            r★ = r★/10
        end
        if !isempty(ie)
            printstyled("Success. ";color = :green)
            println("Existence interval = $ie.")
            fail[k] = false
        else
            printstyled("Failed 🙁\n";color = :red)
            fail[k] = true
        end
        orbit_enc = X .+ Interval(-inf(ie),inf(ie))
        all_rewards = map( j->pendulum_reward_function(orbit_enc[2*j-1:2*j], controller₂(orbit_enc[2*j-1:2*j])  ), 1:m )
        max_reward = round(sup(maximum(all_rewards)),digits=5)
        str_orbit = str_orbit*(method_str*"&"*string(h)*"&"*string(m)*"&"*string(θ₀)*"&"*string(ω₀)*"&"*string(max_reward)*"\\")
    end
    if any(fail)
        failures = sum(fail)
        printstyled("Warning, $failures proof(s) failed. 🙁\n";color=:red)
    end
    return str_orbit
end

function output_LaTeX_table_pendulum(controller_name,csv_string,solutions,orbits,flag,exist_orbit,returns,TP,MR)
    ## DESCRIPTION ##
    # Prints LaTeX string and checks for duplicate orbits.
    # Inputs:
    # controller_name::String
    # csv_string::String -- path to csv file
    # solutions, orbits, flag, exist_orbit,returns,TP,MR(max_reward): these should be outputs from prove_transients_pendulum.
    dataset = DataFrame(CSV.File(csv_string))
    N = length(flag)
    mystring_orbits = ""
    mystring_transients = ""
    distinct_list = Int64[]
    for k=1:N
        if !("explicit" in names(dataset))
            method = "Semi-Implicit"
        elseif dataset.explicit[k] == true
            method = "Explicit"
        else
            method = "Semi-Implicit"
        end
        h = dataset.h[k]
        if exist_orbit[k]
            if flag[k][1:5]=="Fixed"
                exact_step_size = "Yes"
                θ₀ = round(orbits[k][1],digits=5)
                ω₀ = round(orbits[k][2],digits=5)
                m = Int64(length(orbits[k])/2)
            else
                exact_step_size = "No"
                θ₀ = round(orbits[k][2],digits=5)
                ω₀ = round(orbits[k][3],digits=5)
                m = Int64((length(orbits[k])-1)/2)
            end
            max_reward = round(sup(MR[k]),digits=5)
            if occursin("Counter-Clockwise",flag[k])
                Direction="+"
            else
                Direction="-"
            end
            # Check unique
            _,duplicate_list = unique_orbit(orbits,k)
            new = true
            if isempty(distinct_list)
                push!(distinct_list,k)
            elseif !any(map(j->(distinct_list[j] in duplicate_list), 1:length(distinct_list)))
                push!(distinct_list,k)
            else
                new = false
            end
            if new
                mystring_orbits = mystring_orbits*(controller_name*"&"*method*"&"*string(h)*"&"*string(m)*"&"*string(θ₀)*"&"*string(ω₀)*"&"*Direction*"&"*exact_step_size*"&"*string(max_reward)*"\\")
            end
        else
            if flag[k][1:5]=="Fixed"
                exact_step_size = "Yes"
            else
                exact_step_size = "No"
            end
            θ₀ = round(solutions[k][1],digits=5)
            ω₀ = round(solutions[k][2],digits=5)
            return_current = round(Float64(mid(returns[k])),digits=5)
            tp = round(TP[k],digits=5)
            mystring_transients = mystring_transients*(controller_name*"&"*method*"&"*string(h)*"&"*string(θ₀)*"&"*string(ω₀)*"&"*string(tp)*"&"*string(return_current)*"\\")
        end
    end
    return mystring_orbits,mystring_transients
end

function output_LaTeX_table_cartpole(controller_name,csv_string,solutions,penalty,escape_flag)
    ## DESCRIPTION ##
    # Prints LaTeX string.
    # Inputs: 
    # controller_name::String
    # csv_string::String -- path to csv file
    # solutions, penalty, escape_flag : should be outputs from prove_transients_cartpole or prove_transients_cartpole_NeuralNet
    dataset = DataFrame(CSV.File(csv_string))
    N = length(escape_flag)
    mystring = ""
    for k=1:N
        if !("explicit" in names(dataset))
            method = "Semi-Implicit"
        elseif dataset.explicit[k] == true
            method = "Explicit"
        else
            method = "Semi-Implicit"
        end
        h = dataset.h[k]
        x₀ = round(solutions[k][1],digits=3)
        x₀′ = round(solutions[k][2],digits=3)
        θ₀ = round(solutions[k][3],digits=3)
        θ₀′ = round(solutions[k][4],digits=3)
        if escape_flag[k]
            escaped = "Yes"
        else
            escaped = "No"
        end
        pen = round(Float64(mid(penalty[k])),digits=3)
        mystring = mystring*(controller_name*"&"*method*"&"*string(h)*"&"*string(x₀)*"&"*string(x₀′)*"&"*string(θ₀)*"&"*string(θ₀′)*"&"*escaped*"&"*string(pen)*"\\\\")
    end
    return mystring
end