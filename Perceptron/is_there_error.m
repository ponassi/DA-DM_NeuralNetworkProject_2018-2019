function flag = is_there_error(w,b,Y,X,n)
    flag = false;
    
    for j = 1:n
       f = dot(w,X(j,:)) + b;
       if Y(j)*f < 0
           flag = true;
           break;
       end
    end
    
end

