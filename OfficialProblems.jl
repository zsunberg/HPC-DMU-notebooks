module OfficialProblems

using POMDPModels
using GWBeliefMDPs
using PyCall

unshift!(PyVector(pyimport("sys")["path"]), dirname(@__FILE__()))
@pyimport problem_2

export p1, p2, p3

# XXX be sure to use the aa228-2016-project-2 tag of POMDPModels - the tp in the gridworld doesn't work

p1 = GridWorld(rs=[GridWorldState(3,3), GridWorldState(8,8)], rv=[3.,10.], tp=0.6, terminals=Set{GridWorldState}())

p2 = problem_2.env

gw3 = GridWorld(rs=GridWorldState[], rv=Float64[], tp=0.83, terminals=Set{GridWorldState}(), discount_factor=0.99)
cp3 = GWBeliefMDP(GWPOMDP(gw3, 0.1, 6.0))
p3 = DiscreteGWBMDP(cp3)

end
