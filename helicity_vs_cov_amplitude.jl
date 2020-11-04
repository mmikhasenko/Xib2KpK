using ThreeBodyDecay
using Parameters
using LinearAlgebra
using ThreeBodyDecay.PartialWaveFunctions


#                                            _|                
#    _|_|_|    _|_|    _|_|_|      _|_|_|  _|_|_|_|    _|_|_|  
#  _|        _|    _|  _|    _|  _|_|        _|      _|_|      
#  _|        _|    _|  _|    _|      _|_|    _|          _|_|  
#    _|_|_|    _|_|    _|    _|  _|_|_|        _|_|  _|_|_|    


const mp = 0.938
const mK = 0.493
const mXib = 5.797
const ms = ThreeBodyMasses(mK,mp,mK; m0=mXib)
σs0 = randomPoint(ms)

using LinearAlgebra
const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const id  = Matrix(I,2,2)
const id4  = Matrix(I,4,4)
const z = zeros(Int, 2,2);

const γ0 = [id z; z -id]
const γ1 = [z σx; -σx z]
const γ2 = [z σy; -σy z]
const γ3 = [z σz; -σz z];
const γ = (γ1,γ2,γ3,γ0);

sp(a,b) = a[4]*b[4]-sum(a[1:3].*b[1:3])

#                                    _|                                    
#  _|      _|    _|_|      _|_|_|  _|_|_|_|    _|_|    _|  _|_|    _|_|_|  
#  _|      _|  _|_|_|_|  _|          _|      _|    _|  _|_|      _|_|      
#    _|  _|    _|        _|          _|      _|    _|  _|            _|_|  
#      _|        _|_|_|    _|_|_|      _|_|    _|_|    _|        _|_|_|    


function build_four_vectors_rf0(σs)
    @unpack σ1,σ2,σ3 = σs
    @unpack m1,m2,m3,m0 = ms
    # 
    p1mod = sqrt(λ(m0^2,m1^2,σ1))/(2m0)
    p2mod = sqrt(λ(m0^2,m2^2,σ2))/(2m0)
    p3mod = sqrt(λ(m0^2,m3^2,σ3))/(2m0)
    # 
    cθhat12 = cosθhat12(σs,ms^2)
    cθhat23 = cosθhat23(σs,ms^2)
    sθhat23 = sqrt(1-cθhat23^2)
    sθhat12 = sqrt(1-cθhat12^2)
    # 
    p0 = [0,0,0,m0]
    p2 = [0,0,-p2mod,(m0^2+m2^2-σ2)/(2m0)]
    p1 = [-p1mod*sθhat12, 0, -p1mod*cθhat12, (m0^2+m1^2-σ1)/(2m0)]
    p3 = [ p3mod*sθhat23, 0, -p3mod*cθhat23, (m0^2+m3^2-σ3)/(2m0)]
    #
    return p1,p2,p3,p0
end

function build_four_vectors_rf23(σs)
    @unpack σ1,σ2,σ3 = σs
    @unpack m1,m2,m3,m0 = ms
    # 
    p0mod = sqrt(λ(σ1,m0^2,m1^2)/(4σ1))
    p1mod = sqrt(λ(σ1,m0^2,m1^2)/(4σ1))
    p3mod = sqrt(λ(σ1,m2^2,m3^2)/(4σ1))
    p2mod = sqrt(λ(σ1,m3^2,m2^2)/(4σ1))
    # 
    cθ23 = cosθ23(σs,ms^2)
    sθ23 = sqrt(1-cθ23^2)
    # 
    p0 = [0,0,-p0mod,(σ1+m0^2-m1^2)/(2sqrt(σ1))]
    p1 = [0,0,-p1mod,(m0^2-σ1-m1^2)/(2sqrt(σ1))]
    p2 = [ p2mod*sθ23, 0, p2mod*cθ23, (σ1+m2^2-m3^2)/(2sqrt(σ1))]
    p3 = [-p3mod*sθ23, 0,-p3mod*cθ23, (σ1+m3^2-m2^2)/(2sqrt(σ1))]
    #
    return p1,p2,p3,p0
end

function check_four_vectors(p1,p2,p3,p0)
    p1,p2,p3,p0 = build_four_vectors_rf0(σs0)
    @assert invmasssq(p1) ≈ ms.m1^2
    @assert invmasssq(p2) ≈ ms.m2^2
    @assert invmasssq(p3) ≈ ms.m3^2
    @assert invmasssq(p0) ≈ ms.m0^2
    # 
    @assert  prod(p1+p2+p3 - p0 .< 1e-7)
end

check_four_vectors(build_four_vectors_rf0(σs0)...)
check_four_vectors(build_four_vectors_rf23(σs0)...)


#                                                      _|                        _|      
#    _|_|_|    _|_|    _|      _|    _|_|    _|  _|_|        _|_|_|  _|_|_|    _|_|_|_|  
#  _|        _|    _|  _|      _|  _|_|_|_|  _|_|      _|  _|    _|  _|    _|    _|      
#  _|        _|    _|    _|  _|    _|        _|        _|  _|    _|  _|    _|    _|      
#    _|_|_|    _|_|        _|        _|_|_|  _|        _|    _|_|_|  _|    _|      _|_|  


function cov_amplitude1(two_ν,two_λ,p1,p2,p3,p0)
    pK1, pp, pK3 = p1,p2,p3
    σ1,σ2,σ3 = invmasssq(p2+p3), invmasssq(p3+p1), invmasssq(p1+p2)
    @unpack m0,m1,m2,m3 = ms
    #
    ub⁺ = [1; 0; 0; 0] .* sqrt(2m0)
    ub⁻ = [0; 1; 0; 0] .* sqrt(2m0)
    #
    E2 = (m0^2+m2^2-σ2)/(2m0)
    Ep2, Em2 = sqrt(E2+m2), sqrt(E2-m2)
    up⁺ = [0; Ep2; 0; Em2]
    up⁻ = [-Ep2; 0; Em2; 0] 
    # 
    up = two_λ>0 ? up⁺ : up⁻
    ub = two_ν>0 ? ub⁺ : ub⁻
    # 
    term = transpose(up)*γ0 * ( (sp(pK3+pp,γ)+id4*sqrt(σ1)) / (2*sqrt(σ1)) ) * ub
    #
    return term
end
cov_amplitude3(two_ν,two_λ,p1,p2,p3,p0) = cov_amplitude1(two_ν,two_λ,p3,p2,p1,p0)

function cov_amplitude_sumsq(p1,p2,p3,p0)
    pK1, pp, pK3 = p1,p2,p3
    σ1,σ2,σ3 = invmasssq(p2+p3), invmasssq(p3+p1), invmasssq(p1+p2)
    @unpack m0,m1,m2,m3 = ms

    m1 = ( (sp(p3+p2,γ)+id4*sqrt(σ1)) / (2*sqrt(σ1)) )
    m3 = ( (sp(p1+p2,γ)+id4*sqrt(σ3)) / (2*sqrt(σ3)) )
    DMDM = (m1 + m3) * (sp(p0,γ) + id4*m0) * (m1 + m3) * (sp(p2,γ) + id4*m2)
     return sum(diag(DMDM))
end


cov_amplitude_sumsq(build_four_vectors_rf0(σs0)...)
cov_amplitude_sumsq(build_four_vectors_rf23(σs0)...)

cov_amplitude1(ν,λ,σs) = cov_amplitude1(ν,λ, build_four_vectors_rf0(σs)...)
cov_amplitude3(ν,λ,σs) = cov_amplitude3(ν,λ, build_four_vectors_rf0(σs)...)
cov_amplitude(ν,λ,σs) = cov_amplitude1(ν,λ,σs) + cov_amplitude3(ν,λ,σs)
cov_amplitude_sumsq(σs) = cov_amplitude_sumsq(build_four_vectors_rf0(σs)...)



#  _|                  _|  _|            _|    _|                
#  _|_|_|      _|_|    _|        _|_|_|      _|_|_|_|  _|    _|  
#  _|    _|  _|_|_|_|  _|  _|  _|        _|    _|      _|    _|  
#  _|    _|  _|        _|  _|  _|        _|    _|      _|    _|  
#  _|    _|    _|_|_|  _|  _|    _|_|_|  _|      _|_|    _|_|_|  
#                                                            _|  
#                                                        _|_|    


function h1(σs)
    @unpack σ1,σ2,σ3 = σs
    @unpack m0,m1,m2,m3 = ms
    # 
    E2 = (σ1+m2^2-m3^2)/(2sqrt(σ1))
    E0 = (σ1+m0^2-m1^2)/(2sqrt(σ1))
    h = sqrt(E2+m2)*sqrt(E0+m0)
    return h
end

function hel_amplitude1(two_ν,two_λ,σs)
    @unpack σ1,σ2,σ3 = σs
    @unpack m0,m1,m2,m3 = ms
    # 
    h = h1(σs)
    #
    _cosθhat12 = cosθhat12(σs,ms^2)
    _cosθ23 = cosθ23(σs,ms^2)
    _cosζ21_for2 = ThreeBodyDecay.cosζ21_for2(σs,ms^2)
    # 
    angular_term = sum(
        wignerd_doublearg(1,two_ν, two_τ,  _cosθhat12) *
        wignerd_doublearg(1,two_τ, two_λ′, _cosθ23) *
        wignerd_doublearg(1,two_λ′,two_λ,  _cosζ21_for2) * phase(two_λ,two_λ′)
            for two_τ in [-1,1], two_λ′ in [-1,1])
    η = 1.0
    return h * angular_term * η
end

function swap31(σs)
    @unpack σ1,σ2,σ3 = σs
    return Invariants(σ3,σ2,σ1)
end

hel_amplitude3(two_ν,two_λ,σs) = (two_ν==two_λ ? -1 : 1) * hel_amplitude1(two_ν,two_λ,swap31(σs))

hel_amplitude(two_ν,two_λ,σs) = hel_amplitude1(two_ν,two_λ,σs) + hel_amplitude3(two_ν,two_λ,σs)

σs0 = Invariants(ms; σ1 = 10.0, σ3 = 10.0)
σs0 = randomPoint(ms)

sum(abs2, hel_amplitude(two_ν,two_λ,σs0) for two_λ=[-1,1], two_ν=[-1,1])
sum(abs2, cov_amplitude(two_ν,two_λ,σs0) for two_λ=[-1,1], two_ν=[-1,1])
cov_amplitude_sumsq(build_four_vectors_rf0(σs0)...) 
cov_amplitude_sumsq(build_four_vectors_rf23(σs0)...) 
# 
sum(abs2, hel_amplitude1(two_ν,two_λ,σs0) for two_λ=[-1,1], two_ν=[-1,1])
sum(abs2, cov_amplitude1(two_ν,two_λ,σs0) for two_λ=[-1,1], two_ν=[-1,1])
