function accuracy = calculate_accuracy(x_set, y_set, setSize, w, b)
    error = 0;
    for i = 1 : length(y_set)
        f = dot(w, x_set(i, :)) + b;
        if y_set(i) * f < 0
            error = error + 1;
        end
    end

    accuracy = 100 - ((error / (setSize)) * 100);
end

