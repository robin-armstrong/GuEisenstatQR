using GuEisenstatQR
using LinearAlgebra
using Random
using Test

# function to print debug message for a failed test
function showInfo(msg, testResult)
	if(typeof(testResult) != Test.Pass)
		@info msg
	end
end

@testset "test updateRank!" begin
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
	for r = 1:n - k
		gamma_init[r] = norm(C_init[:, r])
	end
	
	omega_init = Vector{Float64}(undef, k)
	for r = 1:k
		omega_init[r] = 1/norm(Ainv_init[r, :])
	end
	
	for j in [1, 3, n - k]
		params_str = "parameters are: j = "*string(j)
		
		A = deepcopy(A_init)
		B = deepcopy(B_init)
		C = deepcopy(C_init)
		Q = deepcopy(Q_init)
		perm = deepcopy(perm_init)
		AinvB = deepcopy(AinvB_init)
		gamma = deepcopy(gamma_init)
		omega = deepcopy(omega_init)
		
		A, B, C, AinvB = GuEisenstatQR.updateRank!(j, Q, A, B, C, perm, AinvB, gamma, omega)
		
		showInfo(params_str, @test size(A) == (k + 1, k + 1))
		showInfo(params_str, @test size(B) == (k + 1, n - k - 1))
		showInfo(params_str, @test size(C) == (m - k - 1, n - k - 1))
		
		R = [A B; zeros(size(C, 1), size(A, 2)) C]
		
		showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
		showInfo(params_str, @test norm(tril(A, -1)) < epsilon)
		showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
		showInfo(params_str, @test perm[k + j] == perm_init[k + 1])
		showInfo(params_str, @test perm[k + 1] == perm_init[k + j])
		
		Ainv_target = inv(A)
		AinvB_target = Ainv_target*B
		
		gamma_target = Vector{Float64}(undef, size(C, 2))
		for r = 1:size(C, 2)
			gamma_target[r] = norm(C[:, r])
		end

		omega_target = Vector{Float64}(undef, size(A, 1))
		for r = 1:size(A, 1)
			omega_target[r] = 1/norm(Ainv_target[r, :])
		end
			
		showInfo(params_str, @test norm(AinvB - AinvB_target)/norm(AinvB_target) < epsilon)
		showInfo(params_str, @test norm(gamma - gamma_target)/norm(gamma_target) < epsilon)
		showInfo(params_str, @test norm(omega - omega_target)/norm(omega_target) < epsilon)
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
	for r = 1:n - k
		gamma_init[r] = norm(C_init[:, r])
	end
	
	omega_init = Vector{Float64}(undef, k)
	for r = 1:k
		omega_init[r] = 1/norm(Ainv_init[r, :])
	end
	
	for i in [1, 3, k]
		for j in [1, 3, n - k]
			params_str = "parameters are: i = "*string(i)*", j = "*string(j)
			
			A = deepcopy(A_init)
			B = deepcopy(B_init)
			C = deepcopy(C_init)
			Q = deepcopy(Q_init)
			perm = deepcopy(perm_init)
			AinvB = deepcopy(AinvB_init)
			gamma = deepcopy(gamma_init)
			omega = deepcopy(omega_init)
			
			GuEisenstatQR.updateFactors!(i, j, Q, A, B, C, perm, AinvB, gamma, omega)
			R = [A B; zeros(m - k, k) C]
			
			showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
			showInfo(params_str, @test norm(tril(A, -1)) < epsilon)
			showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
			showInfo(params_str, @test perm[k + 1] == perm_init[i])
			showInfo(params_str, @test perm[k] == perm_init[k + j])
			
			Ainv_target = inv(A)
			AinvB_target = Ainv_target*B
			
			gamma_target = Vector{Float64}(undef, n - k)
			for r = 1:n - k
				gamma_target[r] = norm(C[:, r])
			end
	
			omega_target = Vector{Float64}(undef, k)
			for r = 1:k
				omega_target[r] = 1/norm(Ainv_target[r, :])
			end
				
			showInfo(params_str, @test norm(AinvB - AinvB_target)/norm(AinvB_target) < epsilon)
			showInfo(params_str, @test norm(gamma - gamma_target)/norm(gamma_target) < epsilon)
			showInfo(params_str, @test norm(omega - omega_target)/norm(omega_target) < epsilon)
		end
	end
end
