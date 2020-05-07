% find phi_x

function sol = get_phi(N,x,sigma,ep)

for i=1:N+1
    if x(i) < sigma
        %sol(i) = 1 - exp(-(1+x(i))/(2*ep))  - (1 - exp(-(1+sigma)/(2*ep)))*(x(i)+1)/((1+sigma));
        sol(i) = 1 - exp(-(1+x(i))/(ep))  - (1 - exp(-(1+sigma)/(ep)))*(x(i)+1)/((1+sigma));
    else
        sol(i) = 0;
    end
end



return