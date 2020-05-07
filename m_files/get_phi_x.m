% find phi_x

function sol = get_phi_x(N,x,sigma,ep)

for i=1:N+1
    if x(i) < sigma
        sol(i) = (exp(-(sigma + 1)/(2*ep)) - 1)/(sigma + 1) + exp(-(x(i) + 1)/(2*ep))/(2*ep);
        %sol(i) = (exp(-(sigma + 1)/ep) - 1)/(sigma + 1) + exp(-(x(i) + 1)/ep)/ep;
    else
        sol(i) = 0;
    end
end

return