module GWBeliefMDPs

importall POMDPs
importall GenerativeModels
using POMDPModels
using POMDPToolbox

export 
    GWBelief,
    GWBeliefMDP,
    DiscreteGWBMDP,
    GWPState,
    GWPOMDP,
    GWUpdater,
    GWBelief,
    generate_s,
    generate_sr,
    generate_o

export
    GoToMostLikely

const RMAP = Dict{Int, Tuple{Int,Int}}(1=>(3,3), 2=>(3,8), 3=>(8,8), 4=>(8,3))
const BACK_RMAP = Dict{Tuple{Int,Int}, Int}((3,3)=>1, (3,8)=>2, (8,8)=>3, (8,3)=>4)
const AMAP = Dict{Int, GridWorldAction}(1=>GridWorldAction(:left),
                                        2=>GridWorldAction(:right),
                                        3=>GridWorldAction(:up),
                                        4=>GridWorldAction(:down))

#= actions
1: left
2: right
3: up
4: down
5: look (3,3)
6: look (3,8)
7: look (8,8)
8: look (8,3)
=#

immutable GWPState
    x::Int
    y::Int
    r::Int
end

type GWPOMDP <: POMDP{GWPState, Int}
    base::GridWorld
    p_change::Float64
    d0::Float64 # half efficiency distance
end
GWPOMDP() = GWPOMDP(GridWorld(), 0.1, 5.0)
discount(p::GWPOMDP) = discount(p.base)

function generate_s(pomdp::GWPOMDP, s::GWPState, a::Int, rng::AbstractRNG, sp=nothing)
    if a <= 4
        gwsp = generate_s(pomdp.base, GridWorldState(s.x,s.y), AMAP[a], rng)
    else
        gwsp = GridWorldState(s.x, s.y)
    end
    if rand(rng) < pomdp.p_change
        r = mod1(s.r + rand(rng, 1:3), 4)
    else
        r = s.r
    end
    return GWPState(gwsp.x, gwsp.y, r)
end

function generate_o(pomdp::GWPOMDP, s::GWPState, a::Int, sp::GWPState, rng::AbstractRNG, o=nothing)
    if a > 4
        checked = a-4
        rxy = RMAP[checked]
        d = sqrt((sp.x-rxy[1])^2 + (sp.y-rxy[2])^2)
        eta = 2.0^-(d/pomdp.d0)
        if sp.r == checked
            r_meas = Nullable{Bool}(rand(rng)<eta || rand(rng)<0.25)
        else
            r_meas = Nullable{Bool}(rand(rng)<(1.0-eta)*0.25)
        end
    else
        r_meas = Nullable{Bool}()
    end
    got_reward = get(BACK_RMAP, (sp.x, sp.y), 0) == sp.r

    return GWObs(sp.x, sp.y, got_reward, r_meas)
end

reward(pomdp::GWPOMDP, s::GWPState, a::Int, sp::GWPState) = reward(pomdp, sp)

function reward(pomdp::GWPOMDP, sp::GWPState)
    if sp.r == get(BACK_RMAP, (sp.x, sp.y), 0)
        return 1
    else
        return 0
    end
end

immutable GWBelief
    x::Int
    y::Int
    b::NTuple{3,Float64}
end
Base.isequal(a::GWBelief, b::GWBelief) = a.x==b.x && a.y==b.y && a.b==b.b

function rand(rng::AbstractRNG, b::GWBelief)
    # sample from b.b
    t = rand(rng)
    i = 1
    cb = b.b[1]
    while cb < t && i < length(b.b)
        i += 1
        @inbounds cb += b.b[i]
    end
    if cb < t
        i += 1
    end

    return GWPState(b.x, b.y, i)
end

function pdf(b::GWBelief, s::GWPState)
    if s.x == b.x && s.y == b.y
        if s.r < 4
            return b.b[s.r]
        else # r = 4
            return 1.0 - sum(b.b)
        end
    else
        return 0.0
    end
end

type GWUpdater <: Updater{GWBelief}
    p::GWPOMDP
    rb::Vector{Float64}
end
GWUpdater(p::GWPOMDP) = GWUpdater(p, Array(Float64,4))

immutable GWObs
    x::Int
    y::Int
    got_reward::Bool
    r_meas::Nullable{Bool}
end

function update(up::GWUpdater, b::GWBelief, a::Int, o::GWObs, bnew=nothing)
    # rb = zeros(4)
    rb = up.rb
    fill!(rb, 0.0)
    b4 = 1.-sum(b.b)
    r = get(BACK_RMAP, (o.x,o.y), 0)
    if o.got_reward
        rb[r] = 1.0
    else
        # step update
        for i in 1:4
            if i <= 3
                rb[i] = (1.-up.p.p_change)*b.b[i]
                rb[i] += up.p.p_change*b4/3.
            else
                rb[i] = (1.-up.p.p_change)*b4
            end
            for j in 1:3
                if j != i
                    rb[i] += up.p.p_change*b.b[j]/3.
                end
            end
        end
        @assert abs(sum(rb)-1.0) < 0.01

        # observation update
        # rb2 = zeros(4)
        if a > 4
            checked = a-4
            rxy = RMAP[checked]
            d = sqrt((o.x-rxy[1])^2 + (o.y-rxy[2])^2)
            eta = 2.0^-(d/up.p.d0)
            if get(o.r_meas)
                for i in 1:4
                    if i == checked
                        rb[i] = (0.75*eta + 0.25)*rb[i]
                    else
                        rb[i] = (0.25 - 0.25*eta)*rb[i]
                    end
                end
            else
                for i in 1:4
                    if i == checked
                        rb[i] = (0.75 - 0.75*eta)*rb[i]
                    else
                        rb[i] = (0.75 + 0.25*eta)*rb[i]
                    end
                end
            end
            rb ./= sum(rb)
        end

        if r > 0 # on a reward square, but did not receive a reward
            rb[r] = 0.0
            rb ./= sum(rb)
        end
    end

    #XXX emergency hack bug fix
    for i in 1:3
        if rb[i] < 0.0
            rb[i] = 0.0
        end
    end
    # println(STDERR, """
    #     b: $b
    #     a: $a
    #     o: $o
    #     """)

    return GWBelief(o.x, o.y, (rb[1],rb[2],rb[3]))
end


type GWBeliefMDP <: MDP{GWBelief, Int}
    base::GWPOMDP
end
GWBeliefMDP() = GWBeliefMDP(GWPOMDP())
discount(p::GWBeliefMDP) = discount(p.base)
create_state(p::GWBeliefMDP) = GWBelief(1,1,(0.0, 0.0, 0.0))

function state_index(mdp::GWBeliefMDP, b::GWBelief)
    si = sub2ind((26, mdp.base.base.size_x, 26, mdp.base.base.size_y, 26),
                round(Int, b.b[1]*25.0) + 1,
                b.x,
                round(Int, b.b[2]*25.0) + 1,
                b.y,
                round(Int, b.b[3]*25.0) + 1
           )
    @assert si > 0
    return si
end

function GWBelief(mdp::GWBeliefMDP, ind::Int)
    sub = ind2sub((26, mdp.base.base.size_x, 26, mdp.base.base.size_y, 26), ind)
    return GWBelief(sub[2], sub[4], ((sub[1]-1)/25, (sub[3]-1)/25, (sub[5]-1)/25))
end

function generate_sr(mdp::GWBeliefMDP, s::GWBelief, a::Int, rng::AbstractRNG, sp=nothing)
    sp = generate_s(mdp, s, a, rng)
    base_sp = rand(rng, sp)
    return (sp, reward(mdp.base, base_sp))
end

function generate_s(mdp::GWBeliefMDP, s::GWBelief, a::Int, rng::AbstractRNG, sp=nothing)
    base_s = rand(rng, s)
    base_sp = generate_s(mdp.base, base_s, a, rng)
    base_o = generate_o(mdp.base, base_s, a, base_sp, rng)
    sp = update(GWUpdater(mdp.base), s, a, base_o)
    return sp
end

function initial_state(mdp::GWBeliefMDP, rng::AbstractRNG)
    # http://math.stackexchange.com/questions/354278/generating-a-random-probability-vector
    b = randexp(rng, 4)
    b ./= sum(b)
    x = rand(rng, 1:mdp.base.base.size_x)
    y = rand(rng, 1:mdp.base.base.size_y)
    return GWBelief(x, y, (b[1], b[2], b[3]))
end


type DiscreteGWBMDP <: MDP{Int, Int}
    base::GWBeliefMDP
end
discount(p::DiscreteGWBMDP) = discount(p.base)
create_state(p::DiscreteGWBMDP) = 0

function generate_sr(mdp::DiscreteGWBMDP, s::Int, a::Int, rng::AbstractRNG, sp=nothing)
    base_sp, r = generate_sr(mdp.base, GWBelief(mdp.base, s), a, rng)
    si = state_index(mdp.base, base_sp)
    return si, r
end

initial_state(mdp::DiscreteGWBMDP, rng::AbstractRNG) = state_index(mdp.base, initial_state(mdp.base, rng))

actions(mdp::Union{GWBeliefMDP,GWPOMDP,DiscreteGWBMDP}) = 1:8

include("heuristics.jl")

end
