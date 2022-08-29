using GuEisenstatQR
using LinearAlgebra
using Random
using Test

# function to print debug messages for failed test
function showInfo(msg, testResult)
	if(typeof(testResult) != Test.Pass)
		@info msg
	end
end

@testset "test updateFactors!" begin
	m = 10
	n = 10
	k = 5
	epsilon = 1e-12
	
	Q_init = Matrix(qr(randn(m, m)).Q)
	R_init = randn(m, n)
	R_init[:, 1:min(m, n)] = triu(R_init[:, 1:min(m, n)])
	perm_init = randperm(n)
	M = Matrix{Float64}(undef, m, n)
	M[:, perm_init] = Q_init*R_init
	
	A_init = R_init[1:k, 1:k]
	B_init = R_init[1:k, k + 1:n]
	C_init = R_init[k + 1:m, k + 1:n]
	Ainv_init = inv(A_init)
	AinvB_init = Ainv_init*B_init
	
	gamma_init = Vector{Float64}(undef, n - k)
	for i = 1:n - k
		gamma_init[i] = norm(C_init[:, i])
	end
	
	omega_init = Vector{Float64}(undef, k)
	for i = 1:k
		omega_init[i] = 1/norm(Ainv_init[i, :])
	end
	
	for i in [1, 3, k]
		for j in [1, 3, 5]
			params_str = "parameters are: i = "*string(i)*", j = "*string(j)
			
			A = A_init
			B = B_init
			C = C_init
			Q = Q_init
			perm = perm_init
			AinvB = AinvB_init
			gamma = gamma_init
			omega = omega_init
			
			GuEisenstatQR.updateFactors!(i, j, k, Q, A, B, C, perm, AinvB, gamma, omega)
			R = [A B; zeros(m - k, k) C]
			
			println("A = ")
			display(A)
			println("")
			println("R = ")
			display(R)
			println("")
			
			showInfo(params_str, @test norm(M[:, perm] - Q*R) < epsilon)
			showInfo(params_str, @test norm(tril(A, -1)) < epsilon)
			showInfo(params_str, @test norm(Q'*Q - I(m)) < epsilon)
			showInfo(params_str, @test perm[k] == perm_init[i])
			showInfo(params_str, @test perm[k + 1] == perm_init[j])
			
			Ainv_target = inv(A)
			AinvB_target = Ainv_target*B
			
			gamma_target = Vector{Float64}(undef, n - k)
			for i = 1:n - k
				gamma_target[i] = norm(C[:, i])
			end
	
			omega_target = Vector{Float64}(undef, k)
			for i = 1:k
				omega_target[i] = 1/norm(Ainv_target[i, :])
			end
			
			showInfo(params_str, @test norm(AinvB - AinvB_target) < epsilon)
			showInfo(params_str, @test norm(gamma - gamma_target) < epsilon)
			showInfo(params_str, @test norm(omega - omega_target) < epsilon)
		end
	end
end
