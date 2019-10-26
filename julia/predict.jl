using DelimitedFiles
using LinearAlgebra
TF=Float64
function readfile(path, n, d)
    #TODO: deduce n & d from the file!

    X = fill(TF(0.0), (n, d))
    Y = fill(Int32(0.0), n)

    linenum = 1;
    open(path) do file
        for line in eachline(file)
            if linenum>n
                break
            end
            fields = split(line)
            Y[linenum] = parse(Int32,fields[1])
            for f in fields[2:end]
                s = split(f,":")
                k = parse(Int,s[1])
                v = parse(TF, s[2])
                X[k, linenum] = v
            end
            linenum+=1
        end
    end
    X,Y
end



function main()
    if length(ARGS) != 3
        println("Usage: ./exe model_file test_file test_size")
        return
    end
    MF = ARGS[1]
    TF = ARGS[2]
    test_size = parse(Int,ARGS[3])
    
    println("Starting")
    A,hdr = readdlm(MF, ',', Float64; header=true)
    n,dp1 = size(A)
    SV = view(A, :, 2:dp1)
    ay = view(A, :, 1)
    b = parse(Float64, hdr[1]);
    gamma = parse(Float64, hdr[2]);
    println("Values initialized")
    function f(x)
        R = 0.0
        for i=1:n
            R += ay[i]*exp(-gamma*norm(SV[i,:] - x)^2)
        end
        R + b
    end
    d = dp1 - 1
    X,Y = readfile(TF, test_size, d)
    println("Reading file completed")
    good = 0
    for i=1:test_size
        if (Y[i] * f(X[:,i])) > 0
           good += 1
        end
    end
    println("Accuracy $(100*good/test_size)%, $(good)/$(test_size)")
            
end

main()
