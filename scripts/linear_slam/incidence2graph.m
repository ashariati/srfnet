function G = incidence2mat(I)

[s, ~] = find(I' == -1);
[t, ~] = find(I' == 1);

G = digraph(s, t);

end
