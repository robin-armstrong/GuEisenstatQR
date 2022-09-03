struct SRRQR
	numrank::Integer
	perm::Vector
	Q::Matrix
	R::Matrix
end

Base.iterate(F::SRRQR) = (F.numrank, Val(:perm))
Base.iterate(F::SRRQR, ::Val{:perm}) = (F.perm, Val(:Q))
Base.iterate(F::SRRQR, ::Val{:Q}) = (F.Q, Val(:R))
Base.iterate(F::SRRQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(F::SRRQR, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::SRRQR)
	summary(io, F); println(io)
	println(io, "numerical rank:")
	show(io, mime, F.numrank)
	println(io, "\npermutation:")
	show(io, mime, F.perm)
	println(io, "\nQ factor:")
	show(io, mime, F.Q)
	println(io, "\nR factor:")
	show(io, mime, F.R)
end
