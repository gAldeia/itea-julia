
using Test


import ITEAregressor


@testset "Generation of ITs.jl" begin
    # It will fail to match a constructor that takes the test parameters
    @test_throws MethodError ITEAregressor.IT(0.5, [1, 2])
    @test_throws MethodError ITEAregressor.IT(identity, [1, 2], 1.0)
    @test_throws MethodError ITEAregressor.IT(identity, [1, 2], 1.0, [1.0])
end