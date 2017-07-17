module QLearn

using DataFrames

export qlearn

"""
Learn a policy from the data in medium.csv.

alpha: learning rate (> 0)
lambda: eligibility trace decay (in [0, 1])
dataruns: number of times to run through the data
"""
function qlearn(;alpha=0.1, lambda=0.0, dataruns=5)
    df = readtable("medium.csv"); 
    
    q_init = 0.0
    n_states = 50000
    n_actions = 7
    n_pos = 500
    n_vel = 100
    gamma = 1.0

    episodes = Any[]

    episode_start = 1

    N = spzeros(n_states, n_actions)
    Q = fill(q_init, (n_states, n_actions))

    for i in 1:nrow(df)-1
        Q[df[i, :s], df[i, :a]] = df[i,:r]
        if df[i+1, :s] != df[i, :sp]
            push!(episodes, df[episode_start:i, :])
            episode_start = i+1
        end
    end
    push!(episodes, df[episode_start:end, :])

    for run in 1:dataruns
        for (k, ep) in enumerate(episodes)
            # if nrow(ep) == 500
            #     print("\rskipped episode $k            ")
            #     continue
            # end
            N = spzeros(n_states, n_actions)
            sv = convert(Vector{Int}, ep[:s])
            av = convert(Vector{Int}, ep[:a])
            rv = convert(Vector{Int}, ep[:r])
            spv = convert(Vector{Int}, ep[:sp])
            for t in 1:nrow(ep)-1
                s = sv[t]
                a = av[t]
                r = rv[t]
                sp = spv[t]
                N[s,a] = N[s,a] + 1
                # delta = r + gamma*maximum(Q[sp, :]) - Q[s, a]
                delta = r + gamma*Q[sp, av[t+1]] - Q[s,a] # sarsa
                ss, aa, NN = findnz(N)
                for i in 1:length(NN)
                    Q[ss[i], aa[i]] += alpha*delta*NN[i]
                    N[ss[i], aa[i]] = gamma*lambda*NN[i]
                end
            end
            print("\rcompleted episode $k of $(length(episodes))")
        end
        println("\rcompleted run $run of $dataruns         ")
    end
    
    policy_vec = Array(Int, n_states)
    policy_vec[1] = 4
    n_nonfour = 0
    for s in 2:length(policy_vec)
        if all(Q[s,:] .== q_init)
            policy_vec[s] = policy_vec[s-1]
        else
            policy_vec[s] = indmax(Q[s,:])
            n_nonfour += 1
        end
    end
    println(n_nonfour)

    Upv = zeros(n_pos, n_vel)
    Ppv = zeros(n_pos, n_vel)
    for s in 1:n_states
        pos = (s-1) % n_pos
        vel = (s-1-pos)/n_pos
        i = round(Int,pos)+1
        j = round(Int,vel)+1
        Upv[i,j] = maximum(Q[s,:])
        Ppv[i,j] = policy_vec[s]
    end
    
    return Upv, policy_vec
end

end
