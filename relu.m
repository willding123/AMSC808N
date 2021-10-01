function [f] = relu(x)
for i=1:6
    if x(i)>0
        f(i)=x(i);
    else
        f(i)=0;
    end
end


end

