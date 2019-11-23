function It = temporalGradient(I0, I)

T0 = single(I0.data);
T0(I0.undefMask) = NaN;
T = single(I.data);
T(I.undefMask) = NaN;

It = T - T0;

end
