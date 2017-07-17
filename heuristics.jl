type GoToMostLikely <: Policy
    mdp::GWBeliefMDP
end

function action(p::GoToMostLikely, i::Int)
    return action(p, GWBelief(p.mdp, i))
end

function action(p::GoToMostLikely, b::GWBelief)
    r = indmax(pdf(b, GWPState(b.x, b.y, r)) for r in 1:4)
    rx, ry = GWBeliefMDPs.RMAP[r]
    if b.x == rx
        if b.y == ry
            return r%4+5 # look around
            # return r + 4 # look at yourself (this is bad)
        elseif b.y < ry
            return 3 # up
        else # b.y > ry
            return 4 # down
        end
    elseif b.x > rx
        return 1 # left
    else # b.x > rx
        return 2 # right
    end
end
